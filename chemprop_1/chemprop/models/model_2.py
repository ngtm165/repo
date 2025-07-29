# modified_mpnn.py

from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_undirected, add_self_loops, k_hop_subgraph

from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch, TrainingBatch
from chemprop.nn import ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch

# ==============================================================================
# ---------- CÁC LỚP PHỤ TRỢ (HELPER COMPONENTS) -------------------------------
# ==============================================================================

# Cần import thêm hàm to_torch_coo_tensor
from torch_geometric.utils import to_undirected, k_hop_subgraph, to_torch_coo_tensor

def add_k_jump_edges(edge_index, num_nodes, k: int = 3):
    """
    Thêm các cạnh ảo k-bước bằng cách sử dụng phép nhân ma trận kề.
    Điều này tương đương với việc thêm các cạnh cho các đường đi có độ dài lên tới k.
    """
    if k <= 1:
        return edge_index
        
    # 1. Chuyển danh sách cạnh sang ma trận kề thưa (sparse tensor)
    adj = to_torch_coo_tensor(edge_index, size=num_nodes).coalesce()
    
    new_edge_list = [edge_index]
    adj_k = adj.clone()

    # 2. Lặp từ 2 đến k để tính A^2, A^3, ..., A^k
    for hop in range(2, k + 1):
        # Nhân ma trận kề với chính nó để tìm các đường đi dài hơn
        adj_k = torch.sparse.mm(adj_k, adj).coalesce()
        
        # Thêm các cạnh mới tìm được vào danh sách
        # adj_k.indices() sẽ có shape [2, số_cạnh_mới]
        new_edge_list.append(adj_k.indices())

    # 3. Nối tất cả các cạnh lại với nhau
    combined_edges = torch.cat(new_edge_list, dim=1)
    
    # 4. Trả về đồ thị vô hướng và loại bỏ các cạnh trùng lặp
    return to_undirected(combined_edges)


class MixHopConv(nn.Module):
    """Khối MixHop: chạy GINConv song song ở các tầm nhìn khác nhau."""
    def __init__(self, hidden, hops=(1, 2, 3)):
        super().__init__()
        self.hops = hops
        
        # # Tạo MLP cho GINConv một cách tường minh hơn
        # conv_mlps = []
        # for _ in hops:
        #     mlp = nn.Sequential(
        #         nn.Linear(hidden, hidden), 
        #         nn.ReLU(), 
        #         nn.Linear(hidden, hidden)
        #     )
        #     conv_mlps.append(GINConv(mlp))
        
        # self.convs = nn.ModuleList(conv_mlps)
        
        # self.out_proj = nn.Linear(hidden * len(hops), hidden)

        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
            for _ in hops
        ])
        self.out_proj = nn.Linear(hidden * len(hops), hidden)

    def forward(self, x, edge_index):
        outs = []
        for hop, conv in zip(self.hops, self.convs):
            h = x
            for _ in range(hop):
                h = conv(h, edge_index)
            outs.append(h)
        x = self.out_proj(torch.cat(outs, dim=-1))
        return x


class GatedSkipBlock(nn.Module):
    """
    Cổng hóa thông tin từ các nguyên tử và tổng hợp chúng cho mỗi phân tử trong batch.
    Lớp này hoạt động trên toàn bộ batch một cách vector hóa.
    """
    def __init__(self, hidden):
        super().__init__()
        # Mạng nơ-ron nhỏ để tính toán "cổng" alpha cho mỗi nguyên tử
        self.gate_nn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        # Lớp linear để biến đổi đặc trưng nguyên tử trước khi nhân với cổng
        self.message_transform = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h_atoms: Tensor, batch_map: Tensor) -> Tensor:
        """
        Args:
            h_atoms: Tensor đặc trưng của tất cả các nguyên tử trong batch [N_total_atoms, hidden_size].
            batch_map: Vector ánh xạ mỗi nguyên tử tới chỉ số phân tử của nó [N_total_atoms].

        Returns:
            Tensor tin nhắn tổng hợp cho mỗi phân tử trong batch [batch_size, hidden_size].
        """
        # 1. Tính toán cổng alpha cho mỗi nguyên tử
        alpha = torch.sigmoid(self.gate_nn(h_atoms))  # Shape: [N_total_atoms, 1]

        # 2. Tạo tin nhắn từ mỗi nguyên tử
        messages = self.message_transform(h_atoms)  # Shape: [N_total_atoms, hidden_size]

        # 3. Áp dụng cổng vào tin nhắn
        gated_messages = alpha * messages  # Shape: [N_total_atoms, hidden_size]

        # 4. Tổng hợp các tin nhắn theo từng phân tử
        aggregated_messages = global_add_pool(gated_messages, batch_map) # Shape: [batch_size, hidden_size]

        return aggregated_messages


# ==============================================================================
# ---------- LỚP MPNN ĐÃ ĐƯỢC CHỈNH SỬA VÀ HOÀN THIỆN --------------------------
# ==============================================================================

class MPNN_Modified(pl.LightningModule):
    """
    Phiên bản MPNN nâng cao, tích hợp các ý tưởng:
    - K-Jump Rewiring
    - MixHop Block
    - Global Super-node 'S'
    - Gated Skip Connections
    - Jump Knowledge
    """
    def __init__(
        self,
        message_passing: MessagePassing,
        predictor: Predictor,
        k_jump: int = 3,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "predictor"])
        hidden_dim = message_passing.output_dim

        # Các khối kiến trúc
        self.message_passing = message_passing
        self.mixhop = MixHopConv(hidden_dim)
        self.skip_block = GatedSkipBlock(hidden_dim)
        self.predictor = predictor
        
        # Các tham số và cấu hình
        self.k_jump = k_jump
        self.s_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.s_gru = nn.GRUCell(hidden_dim, hidden_dim) # GRU cell để cập nhật trạng thái siêu nút
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        # Cấu hình Metrics và Tốc độ học
        self.metrics = nn.ModuleList([*metrics, self.criterion.clone()]) if metrics else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        self.warmup_epochs, self.init_lr, self.max_lr, self.final_lr = warmup_epochs, init_lr, max_lr, final_lr

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion
        
    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    def fingerprint(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
        """Luồng tính toán 'dấu vân tay' đã được sửa lỗi và vector hóa."""
        # 1. Lớp MPNN ban đầu để lấy đặc trưng cục bộ
        h_mp = self.message_passing(bmg, V_d)

        # 2. Lớp MixHop để lấy đặc trưng đa quy mô
        # edge_index_k_jump = add_k_jump_edges(bmg.edge_index, bmg.n_atoms, k=self.k_jump)
        edge_index_k_jump = add_k_jump_edges(bmg.edge_index, bmg.V.shape[0], k=self.k_jump)

        h_mixhop = self.mixhop(h_mp, edge_index_k_jump)

        # 3. Jump Knowledge: Kết hợp các đặc trưng
        h_atoms = h_mp + h_mixhop

        # 4. Lấy tin nhắn tổng hợp từ các nguyên tử thông qua GatedSkipBlock
        aggregated_atom_messages = self.skip_block(h_atoms, bmg.batch)

        # 5. Cập nhật trạng thái siêu nút S
        num_mols = len(bmg)
        s_state = self.s_init.repeat(num_mols, 1) # Trạng thái ban đầu của siêu nút cho cả batch
        s_state_updated = self.s_gru(aggregated_atom_messages, s_state) # Cập nhật trạng thái
        
        # 6. Dấu vân tay cuối cùng chính là trạng thái của các siêu nút
        H = s_state_updated
        
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
        """Quá trình truyền thẳng."""
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = len(batch[0])
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        loss = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        
        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()
    
    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        # Log validation loss separately if needed
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor(Z) # Use forward for eval
        
        # Use the last metric, which is the criterion clone
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = len(batch[0])
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        
        # For evaluation, don't use additional scaling on weights
        weights = torch.ones_like(weights)

        # Log all metrics except the last one (which is the loss function itself)
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size, on_step=False, on_epoch=True)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, V_d, X_d, *_ = batch
        return self(bmg, V_d, X_d)
    
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        
        # Gracefully handle case where trainer is not available
        if self.trainer is None:
            return {"optimizer": opt}

        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
            
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.trainer.max_epochs == -1:
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch
            
        lr_sched = build_NoamLike_LRSched(opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

    # ----- Các phương thức tiện ích và tải mô hình -----
    
    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(
                f"{traceback.format_exc()}\nModel loading failed! It's possible this checkpoint "
                "was generated in v2.0 and needs to be converted to v2.1.\nPlease run "
                f"'chemprop convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint."
            )
            raise
        
        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        # Rebuild metrics if necessary
        if hparams.get("metrics") is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults") or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]
        
        # Rebuild criterion if necessary
        if hparams.get("predictor", {}).get("criterion") is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (not torch.cuda.is_available() and metric.device.type != "cpu"):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        # Rebuild submodules from hparams if not provided
        # NOTE: Removed 'agg' as it's no longer a direct component
        submodules |= {
            key: Factory.build(hparams[key].pop("cls"), **hparams[key])
            for key in ("message_passing", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs) -> "MPNN_Modified":
        # NOTE: Removed 'agg' from submodule check
        submodules = {k: v for k, v in kwargs.items() if k in ["message_passing", "predictor"]}
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        # The following logic with buffer is a workaround for a PyTorch Lightning loading issue
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location=map_location, hparams_file=hparams_file, strict=strict, **kwargs)