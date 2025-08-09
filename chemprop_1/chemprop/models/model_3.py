from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim
from torch_geometric.nn import global_add_pool

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

class GatedSkipBlock(nn.Module):
    """
    Cổng hóa thông tin từ các nguyên tử và tổng hợp chúng cho mỗi phân tử trong batch.
    """
    def __init__(self, hidden):
        super().__init__()
        self.gate_nn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.message_transform = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h_atoms: Tensor, batch_map: Tensor) -> Tensor:
        alpha = torch.sigmoid(self.gate_nn(h_atoms))
        messages = self.message_transform(h_atoms)
        gated_messages = alpha * messages
        aggregated_messages = global_add_pool(gated_messages, batch_map)
        return aggregated_messages

# ==============================================================================
# ---------- LỚP MPNN ĐÃ ĐƯỢC CHỈNH SỬA VÀ HOÀN THIỆN --------------------------
# ==============================================================================

class MPNN_Simple(pl.LightningModule):
    """
    Phiên bản MPNN tinh gọn, chỉ bao gồm Message Passing
    và cơ chế Gated Readout (GatedSkipBlock + GRU).
    """
    def __init__(
        self,
        message_passing: MessagePassing,
        predictor: Predictor,
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
        self.skip_block = GatedSkipBlock(hidden_dim)
        self.predictor = predictor
        
        # Các tham số cho siêu nút
        self.s_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.s_gru = nn.GRUCell(hidden_dim, hidden_dim)
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
        # 1. Chạy Message Passing để học đặc trưng nguyên tử
        h_atoms = self.message_passing(bmg, V_d)
    
        # 2. Lấy tin nhắn tổng hợp từ các nguyên tử thông qua GatedSkipBlock
        aggregated_atom_messages = self.skip_block(h_atoms, bmg.batch)

        # 3. Cập nhật trạng thái siêu nút S
        num_mols = len(bmg)
        s_state = self.s_init.repeat(num_mols, 1)
        s_state_updated = self.s_gru(aggregated_atom_messages, s_state)
        
        # 4. Dấu vân tay cuối cùng chính là trạng thái của các siêu nút
        H = s_state_updated
        
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
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

        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor(Z)
        
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
        
        weights = torch.ones_like(weights)

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size, on_step=False, on_epoch=True)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, V_d, X_d, *_ = batch
        return self(bmg, V_d, X_d)
    
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        
        if self.trainer is None: return {"optimizer": opt}
        if self.trainer.train_dataloader is None: self.trainer.estimated_stepping_batches
            
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.trainer.max_epochs == -1: cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch
            
        lr_sched = build_NoamLike_LRSched(opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

    # ----- Các phương thức tiện ích và tải mô hình -----
    @classmethod
    def _load(cls, path, map_location, **submodules):
        try: d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(f"{traceback.format_exc()}\nModel loading failed!...")
            raise
        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError: raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")
        if hparams.get("metrics") is not None:
            hparams["metrics"] = [cls._rebuild_metric(m) for m in hparams["metrics"]]
        if hparams.get("predictor", {}).get("criterion") is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (not torch.cuda.is_available() and metric.device.type != "cpu"):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)
        submodules |= {k: Factory.build(hparams[k].pop("cls"), **hparams[key]) for k in ("message_passing", "predictor") if k not in submodules}
        return submodules, state_dict, hparams

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs) -> "MPNN_Simple":
        submodules = {k: v for k, v in kwargs.items() if k in ["message_passing", "predictor"]}
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)
        return super().load_from_checkpoint(buffer, map_location=map_location, hparams_file=hparams_file, strict=strict, **kwargs)
    
    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model