import os
from email import utils
from typing import Any, Dict, Optional
from lightning_fabric.utilities.types import _PATH

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import utils
import pytorch_lightning as pl
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
# from pl_bolts.callbacks import ORTCallback


# Crutch for PyTorch Lightning to stop logging useless metrics
class _Wandblogger(WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        metrics.pop('epoch', None)
        metrics.pop('global_step', None)
        super().log_metrics(metrics=metrics, step=step)


class LogCodeAndConfigCallback(pl.Callback):
    def __init__(self, logger_params) -> None:
        super().__init__()
        self.logger_params = logger_params

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.experiment.log_code(
            root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        trainer.logger.experiment.config.update({**self.logger_cfg}, allow_val_change=True)


# class SaveTorchScriptCheckpointIO(CheckpointIO):
#     def __init__(self, pl_module: str, pl_module_params: dict, dummy_input) -> None:
#         super().__init__()
#         print(pl_module, pl_module_params)
#         self.pl_module:pl.LightningModule = utils.get_obj(pl_module)(pl_module_params)
#         self.dummy_input = dummy_input

#     def save_checkpoint(self, checkpoint, path, storage_options=None):
#         # Save jit
#         self.pl_module.load_state_dict(checkpoint['state_dict'])
#         traced = torch.jit.trace(
#             self.pl_module.model, self.dummy_input, strict=False
#         )
#         traced.save(path.replace('.ckpt', '_jit.pt'))
#         # Save .ckpt
#         super().save_checkpoint(checkpoint, path, storage_options=storage_options)
    
#     def load_checkpoint(self, path, map_location: Any | None = None) -> Dict[str, Any]:
#         return super().load_checkpoint(path, map_location)

#     def remove_checkpoint(self, path) -> None:
#         super().remove_checkpoint(path.replace('.ckpt', '_jit.pt'))
#         super().remove_checkpoint(path)
        

def train(cfg: dict):
    if not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
    logger = _Wandblogger(**cfg.logger_params) if not cfg.no_logging else None
    callbacks = [
        utils.get_obj(callback.callback)(
            **callback.callback_params | ({"dirpath": cfg.experiment_path} if callback.callback.endswith("ModelCheckpoint") else {})
        ) 
        for callback in cfg.trainer_callbacks
    ]
    if not cfg.no_logging:
        callbacks.append(LogCodeAndConfigCallback(cfg))
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        # plugins=[SaveTorchScriptCheckpointIO(
        #     cfg.lightning_module,
        #     cfg.lightning_module_params,
        #     utils.get_obj(cfg.datamodule_params.dataset).dummy_input()
        # )],
        **cfg.trainer_params
    )
    trainer.fit(
        utils.get_obj(cfg.lightning_module)(cfg.lightning_module_params),
        datamodule=DataModule(cfg.mode, **cfg.datamodule_params), 
        ckpt_path=cfg.resume_path
    )
