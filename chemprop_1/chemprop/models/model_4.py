from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim
from torch_geometric.nn import GINConv, global_add_pool

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

class MixHopConv(nn.Module):
    """
    Khối MixHop: chạy GINConv song song ở các tầm nhìn (hop) khác nhau để học
    các đặc trưng đa quy mô.
    """
    def __init__(self, hidden, hops=(1, 2, 3)):
        super().__init__()
        self.hops = hops
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

# ==============================================================================
# ---------- LỚP MPNN (DMPNN + MIXHOP + GLOBAL POOLING) ------------------------
# ==============================================================================

class MPNN_MixHop_Pool(pl.LightningModule):
    """
    Phiên bản MPNN kết hợp DMPNN và MixHop, sử dụng global pooling để tổng hợp.
    - DMPNN: Lõi truyền tin cơ bản.
    - MixHop: Bổ sung các đặc trưng đa quy mô.
    - Global Add Pooling: Cơ chế readout đơn giản để tạo fingerprint cho đồ thị.
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
        self.mixhop = MixHopConv(hidden_dim)
        self.predictor = predictor
        
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
        # 1. Chạy Message Passing (DMPNN)
        h_mp = self.message_passing(bmg, V_d)

        # 2. Chạy MixHop để lấy đặc trưng đa quy mô
        h_mixhop = self.mixhop(h_mp, bmg.edge_index)

        # 3. Kết hợp đặc trưng từ DMPNN và MixHop
        h_atoms = h_mp + h_mixhop

        # 4. Tổng hợp các đặc trưng nút thành đặc trưng đồ thị bằng global pooling
        H = global_add_pool(h_atoms, bmg.batch)
        
        # 5. Kết hợp với các đặc trưng bổ sung (nếu có)
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

        if hparams.get("metrics") is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults") or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]
        
        if hparams.get("predictor", {}).get("criterion") is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (not torch.cuda.is_available() and metric.device.type != "cpu"):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

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
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs) -> "MPNN_MixHop_Pool":
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