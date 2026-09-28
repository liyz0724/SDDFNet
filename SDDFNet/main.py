"""Hydra entry point for training, validation, testing, and prediction."""
import json
from pathlib import Path

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode not in {"train", "validate", "test", "predict"}:
        raise ValueError("mode must be train, validate, test, or predict")
    if cfg.mode != "train" and not cfg.ckpt_path:
        raise ValueError("ckpt_path is required for evaluation and prediction")
    if cfg.ckpt_path and not Path(cfg.ckpt_path).is_file():
        raise FileNotFoundError(cfg.ckpt_path)
    if cfg.mode in {"test", "predict", "validate"} and cfg.trainer.devices != 1:
        raise ValueError("Use trainer.devices=1 for evaluation to avoid distributed sampler padding")
    pl.seed_everything(cfg.seed, workers=True)
    # A full checkpoint provides encoder weights; no separate download is needed.
    if cfg.ckpt_path:
        cfg.network.model.pretrained = False
        if "ckpt_path" in cfg.network.model:
            cfg.network.model.ckpt_path = None
    data_module = hydra.utils.instantiate(cfg.data)
    network = hydra.utils.instantiate(cfg.network)
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if cfg.mode == "train":
        trainer.fit(network, datamodule=data_module, ckpt_path=cfg.ckpt_path)
        # Distributed training ends here; evaluate the selected file in a separate process.
        if trainer.world_size > 1:
            if trainer.is_global_zero:
                print("Evaluate with mode=test, trainer.devices=1 and the best checkpoint path.")
            return
        best = trainer.checkpoint_callback.best_model_path
        if not best:
            if cfg.test_after_fit or cfg.predict_after_fit:
                raise RuntimeError("No validation-selected checkpoint was saved")
            return
        if cfg.test_after_fit:
            results = trainer.test(network, datamodule=data_module, ckpt_path=best)
            Path(logger.log_dir, "test_metrics.json").write_text(json.dumps(results, indent=2))
        if cfg.predict_after_fit:
            trainer.predict(network, datamodule=data_module, ckpt_path=best, return_predictions=False)
    elif cfg.mode == "validate":
        trainer.validate(network, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "test":
        results = trainer.test(network, datamodule=data_module, ckpt_path=cfg.ckpt_path)
        Path(logger.log_dir, "test_metrics.json").write_text(json.dumps(results, indent=2))
    else:
        trainer.predict(network, datamodule=data_module, ckpt_path=cfg.ckpt_path, return_predictions=False)


if __name__ == "__main__":
    main()
