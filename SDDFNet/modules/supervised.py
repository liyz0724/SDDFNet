from typing import Any, Tuple, Dict, Union, Optional, Type
import os
import cv2
from lightning.pytorch import LightningModule
import torch
import torchmetrics
from legacy.refactor_utils import get_class_weights

class SLModule(LightningModule):
    """Supervised learning module for xBD / xView2 building damage assessment."""

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, train_metric: Type[torchmetrics.Metric], val_metric: Type[torchmetrics.Metric], test_metric: Type[torchmetrics.Metric], optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, model_dir: str, name: str, class_weights: Optional[str]=None, pretrain_path: Optional[str]=None) -> None:
        super().__init__()
        self.model = model
        if pretrain_path is not None:
            print(f'Loading model from {pretrain_path}')
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
            incompatible = self.model.load_state_dict(state_dict, strict=True)
            print(incompatible)
            assert len(state_dict) > 0, 'No parameters loaded'
        self.loss_fn = loss_fn
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.name = name
        self.class_weights = get_class_weights(class_weights) if class_weights is not None else None
        if self.class_weights is not None:
            print(f"[INFO] Using class_weights='{class_weights}': {self.class_weights}", flush=True)
        else:
            print('[INFO] class_weights is None. Using unweighted loss.', flush=True)
        self.val_class_metrics = torch.nn.ModuleList([torchmetrics.F1Score(task='binary') for _ in range(5)])
        self.val_loss_tracker = torchmetrics.MeanMetric()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'train_metric', 'val_metric', 'test_metric', 'optimizer', 'scheduler'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute loss in FP32 for numerical stability.
        """
        y_hat = y_hat.float()
        y = y.float()
        if self.class_weights is not None:
            loss_dict = {}
            for i in range(y_hat.shape[1]):
                loss_dict[f'channel_{i}'] = self.loss_fn(y_hat[:, i, ...], y[:, i, ...])
            loss = torch.stack([loss_dict[f'channel_{i}'] * self.class_weights[i] for i in range(y_hat.shape[1])], dim=0).sum()
            return loss
        return self.loss_fn(y_hat, y)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, Any, str]], batch_idx: int) -> torch.Tensor:
        x: torch.Tensor = batch['img']
        y: torch.Tensor = batch['msk']
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y)
        if not torch.isfinite(loss):
            if self.trainer.is_global_zero:
                print('[ERROR] Non-finite training loss detected.', flush=True)
                print(f'y_hat finite: {torch.isfinite(y_hat).all().item()}', flush=True)
                print(f'y_hat min: {torch.nan_to_num(y_hat).min().item():.6f}', flush=True)
                print(f'y_hat max: {torch.nan_to_num(y_hat).max().item():.6f}', flush=True)
            raise RuntimeError('Non-finite training loss')
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        self.train_metric.update(torch.sigmoid(y_hat), y.int())
        self.log(f'train_{self.train_metric.__class__.__name__}', self.train_metric, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, Any, str]], batch_idx: int) -> torch.Tensor:
        x: torch.Tensor = batch['img']
        y: torch.Tensor = batch['msk']
        y_hat = self.forward(x)
        if not torch.isfinite(y_hat).all():
            if self.trainer.is_global_zero:
                print('[ERROR] Non-finite validation prediction detected.', flush=True)
            raise RuntimeError('Non-finite validation prediction')
        loss = self._compute_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x.shape[0])
        self.val_loss_tracker.update(loss.detach(), weight=x.shape[0])
        self.val_metric.update(torch.sigmoid(y_hat), y.int())
        self.log(f'val_{self.val_metric.__class__.__name__}', self.val_metric, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=x.shape[0])
        y_sigm = torch.sigmoid(y_hat)
        if batch_idx == 0 and self.current_epoch == 0 and self.trainer.is_global_zero and (not self.trainer.sanity_checking):
            print(f'\n[DEBUG] Mean predicted probability per channel: {y_sigm.mean(dim=(0, 2, 3)).detach().cpu().tolist()}', flush=True)
            diff_mean = torch.abs(x[:, 3:, :, :] - x[:, :3, :, :]).mean()
            print(f'[DEBUG] Mean absolute pre/post image difference: {diff_mean.detach().item():.6f}', flush=True)
        loc_pred = y_sigm[:, 0, ...]
        loc_msk = loc_pred > 0.38
        dmg_msk = y_sigm[:, 1:, ...].argmax(dim=1) + 1
        dmg_msk = dmg_msk * loc_msk
        hot_dmg_msk = torch.zeros_like(y_hat)
        for i in range(5):
            hot_dmg_msk[:, i, ...] = dmg_msk == i
        hot_dmg_msk[:, 0, ...] = loc_msk
        for i in range(5):
            self.val_class_metrics[i].update(hot_dmg_msk[:, i, ...].int(), y[:, i, ...].int())
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Print and log xBD-style validation metrics at the end of every epoch.

        We compute:
        - F1loc
        - F1no
        - F1min
        - F1maj
        - F1des
        - F1cls = harmonic mean of four damage-class F1 scores
        - xBD Score = 0.3 * F1loc + 0.7 * F1cls
        """
        avg_val_loss = self.val_loss_tracker.compute()
        f1_loc = self.val_class_metrics[0].compute()
        f1_nodmg = self.val_class_metrics[1].compute()
        f1_minor = self.val_class_metrics[2].compute()
        f1_major = self.val_class_metrics[3].compute()
        f1_destroy = self.val_class_metrics[4].compute()
        eps = 1e-06
        f1_cls = 4.0 / (1.0 / (f1_nodmg + eps) + 1.0 / (f1_minor + eps) + 1.0 / (f1_major + eps) + 1.0 / (f1_destroy + eps))
        xbd_score = 0.3 * f1_loc + 0.7 * f1_cls
        self.log('val_xbd_score', xbd_score, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_F1loc', f1_loc, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_F1cls', f1_cls, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_F1no', f1_nodmg, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_F1min', f1_minor, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_F1maj', f1_major, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_F1des', f1_destroy, prog_bar=False, logger=True, sync_dist=True)
        if self.trainer.is_global_zero and (not self.trainer.sanity_checking):
            print(f'\n[{self.__class__.__name__}] Validation Report - Epoch {self.current_epoch}', flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Metric':<30} | {'Value':<33} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Validation Loss':<30} | {avg_val_loss.item():<33.4f} |", flush=True)
            print(f"| {'Overall xBD Score':<30} | {xbd_score.item():<33.4f} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Localization F1':<30} | {f1_loc.item():<33.4f} |", flush=True)
            print(f"| {'Damage F1 / F1cls':<30} | {f1_cls.item():<33.4f} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'No Damage F1':<30} | {f1_nodmg.item():<33.4f} |", flush=True)
            print(f"| {'Minor F1':<30} | {f1_minor.item():<33.4f} |", flush=True)
            print(f"| {'Major F1':<30} | {f1_major.item():<33.4f} |", flush=True)
            print(f"| {'Destroyed F1':<30} | {f1_destroy.item():<33.4f} |", flush=True)
            print('-' * 72 + '\n', flush=True)
        self.val_loss_tracker.reset()
        for metric in self.val_class_metrics:
            metric.reset()

    def on_predict_start(self) -> None:
        os.makedirs(os.path.join(self.model_dir, f'{self.name}', 'submission'), exist_ok=True)

    @staticmethod
    def pred_one_image(pred: torch.Tensor):
        y_sigm = torch.sigmoid(pred)
        y_pred = y_sigm.cpu().numpy().transpose(1, 2, 0)
        loc_pred = y_pred[..., 0]
        loc_msk = (loc_pred > 0.38).astype('uint8')
        dmg_msk = y_pred[..., 1:].argmax(axis=2) + 1
        dmg_msk = dmg_msk * loc_msk
        loc_msk = loc_msk.astype('uint8')
        dmg_msk = dmg_msk.astype('uint8')
        return (loc_msk, dmg_msk)

    def predict_step(self, batch: Dict[str, Union[torch.Tensor, Any, str]], batch_idx: int, dataloader_idx: int=0) -> Any:
        x, fns = (batch['img'], batch['fn'])
        y_hat = self.forward(x)
        for pred, fn in zip(y_hat.unbind(dim=0), fns):
            file_name = os.path.basename(os.path.dirname(os.path.dirname(fn))) + '__' + os.path.basename(fn)
            loc_msk, msk_dmg = self.pred_one_image(pred)
            cv2.imwrite(os.path.join(self.model_dir, f'{self.name}', 'submission', file_name.replace('_pre_disaster', '_localization_disaster_prediction')), loc_msk)
            cv2.imwrite(os.path.join(self.model_dir, f'{self.name}', 'submission', file_name.replace('_pre_disaster', '_damage_disaster_prediction')), msk_dmg)

    def test_step(self, batch: Dict[str, Union[torch.Tensor, Any, str]], batch_idx: int) -> torch.Tensor:
        x, y = (batch['img'], batch['msk'])
        y_hat = self.forward(x)
        y_sigm = torch.sigmoid(y_hat)
        loc_pred = y_sigm[:, 0, ...]
        loc_msk = loc_pred > 0.38
        dmg_msk = y_sigm[:, 1:, ...].argmax(dim=1) + 1
        dmg_msk = dmg_msk * loc_msk
        hot_dmg_msk = torch.zeros(y_hat.shape, dtype=y_hat.dtype, device=y_hat.device)
        for i in range(5):
            hot_dmg_msk[:, i, ...] = dmg_msk == i
        hot_dmg_msk[:, 0, ...] = loc_msk
        for i in range(y_hat.shape[1]):
            self.test_metric[i].update(hot_dmg_msk[:, i, ...].int(), y[:, i, ...].int())
            self.log(f'test_channel_{i}_F1', self.test_metric[i], prog_bar=True, logger=True, sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        return y_hat

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Support both original optimizer setup and model-defined parameter groups.

        If the model implements get_param_groups(), use it. This enables
        differential learning rates for encoder / decoder / BDA / head.
        Otherwise, fall back to the original baseline behavior.
        """
        if hasattr(self.model, 'get_param_groups'):
            params = self.model.get_param_groups()
        else:
            params = self.model.parameters()
        optimizer = self.optimizer(params)
        scheduler = self.scheduler(optimizer)
        return ([optimizer], [scheduler])

    def on_test_epoch_end(self) -> None:
        """

        Print xBD-style test metrics after testing.

        channel 0: localization

        channel 1: no damage

        channel 2: minor

        channel 3: major

        channel 4: destroyed

        """
        f1_loc = self.test_metric[0].compute()
        f1_nodmg = self.test_metric[1].compute()
        f1_minor = self.test_metric[2].compute()
        f1_major = self.test_metric[3].compute()
        f1_destroy = self.test_metric[4].compute()
        eps = 1e-06
        f1_cls = 4.0 / (1.0 / (f1_nodmg + eps) + 1.0 / (f1_minor + eps) + 1.0 / (f1_major + eps) + 1.0 / (f1_destroy + eps))
        xbd_score = 0.3 * f1_loc + 0.7 * f1_cls
        self.log('test_xbd_score', xbd_score, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_F1loc', f1_loc, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_F1cls', f1_cls, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_F1no', f1_nodmg, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_F1min', f1_minor, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_F1maj', f1_major, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_F1des', f1_destroy, prog_bar=False, logger=True, sync_dist=True)
        if self.trainer.is_global_zero:
            print(f'\n[{self.__class__.__name__}] Test Report', flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Metric':<30} | {'Value':<33} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Overall xBD Score':<30} | {xbd_score.item():<33.4f} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'Localization F1':<30} | {f1_loc.item():<33.4f} |", flush=True)
            print(f"| {'Damage F1 / F1cls':<30} | {f1_cls.item():<33.4f} |", flush=True)
            print('-' * 72, flush=True)
            print(f"| {'No Damage F1':<30} | {f1_nodmg.item():<33.4f} |", flush=True)
            print(f"| {'Minor F1':<30} | {f1_minor.item():<33.4f} |", flush=True)
            print(f"| {'Major F1':<30} | {f1_major.item():<33.4f} |", flush=True)
            print(f"| {'Destroyed F1':<30} | {f1_destroy.item():<33.4f} |", flush=True)
            print('-' * 72 + '\n', flush=True)
        for metric in self.test_metric:
            metric.reset()
