from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch, TrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch

# --- Hàm trợ giúp mới để tạo Lớp Liền kề Hoàn toàn ---
def to_fully_connected(num_nodes: int, device: torch.device) -> Tensor:
    """
    Tạo một chỉ số cạnh được kết nối đầy đủ cho một số lượng nút nhất định.
    """
    adj = torch.ones(num_nodes, num_nodes, device=device) - torch.eye(num_nodes, device=device)
    return adj.nonzero().t()

# --- Lớp bao bọc mới cho MessagePassing ---
class DMPNNWithFA(MessagePassing):
    """
    Một lớp bao bọc cho DMPNN để thêm một Lớp Liền kề Hoàn toàn ở cuối.
    """
    def __init__(self, message_passing: MessagePassing):
        super().__init__()
        self.message_passing = message_passing
        # Lớp truyền tin cuối cùng cho bước FA
        # Kích thước đầu vào và đầu ra giống như hidden_size của DMPNN
        hidden_size = self.message_passing.hidden_size
        self.fa_layer = nn.Linear(hidden_size, hidden_size)
        self.hparams = {"wrapped_dmpnn": message_passing.hparams}
        self.output_dim = self.message_passing.output_dim

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None) -> Tensor:
        # Chạy các lớp DMPNN ban đầu
        # Chúng ta cần lấy biểu diễn nút từ lớp gần cuối
        *_, h_mid = self.message_passing(bmg, V_d, return_hidden=True)
        
        # Tạo chỉ số cạnh được kết nối đầy đủ
        fa_edge_index = to_fully_connected(h_mid.size(0), h_mid.device)
        
        # Áp dụng lớp truyền tin FA
        # Vì đây là một đồ thị hoàn chỉnh, chúng ta có thể đơn giản hóa việc truyền tin
        # bằng cách tổng hợp tất cả các nút và áp dụng một phép biến đổi tuyến tính.
        # Đây là một cách đơn giản hóa, các phương pháp phức tạp hơn có thể được sử dụng.
        
        # Tổng hợp thông điệp từ tất cả các nút khác
        fa_messages = torch.sum(h_mid, dim=0) - h_mid
        
        # Cập nhật biểu diễn nút bằng lớp FA
        h_final = self.fa_layer(fa_messages) + h_mid # Kết nối phần còn lại
        
        # Trả về các biểu diễn đầu ra cuối cùng
        return self.message_passing.W_o(h_final)

class MPNN_1(pl.LightningModule):
    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
        use_fa_layer: bool = False,  # Cờ mới
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        
        # --- Sửa đổi: Bọc message_passing nếu cần ---
        if use_fa_layer and isinstance(message_passing, DMPNN):
            logger.info("Enabling Fully-Adjacent layer for the final message passing step.")
            self.message_passing = DMPNNWithFA(message_passing)
        else:
            self.message_passing = message_passing
            if use_fa_layer:
                logger.warning("`use_fa_layer` is True, but the message passing block is not a supported type (DMPNN). Ignoring.")
        # --- Kết thúc sửa đổi ---

        self.hparams.update(
            {
                "message_passing": self.message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()
        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        
    # ... (phần còn lại của lớp MPNN không thay đổi) ...
    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        if hasattr(self.message_passing, 'V_d_transform'):
            self.message_passing.V_d_transform.train()
        if hasattr(self.message_passing, 'graph_transform'):
            self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, V_d, X_d, *_ = batch

        return self(bmg, V_d, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    def get_batch_size(self, batch: TrainingBatch) -> int:
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(
                f"{traceback.format_exc()}\nModel loading failed (full stacktrace above)! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'chemprop convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint."
            )
        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")
        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN_1:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN_1:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model