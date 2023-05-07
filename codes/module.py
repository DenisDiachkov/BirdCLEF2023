from typing import Any
import torch
import utils
from multipledispatch import dispatch
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod


class BaseModule(LightningModule, ABC):
    @dispatch(torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.nn.Module)
    def __init__(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
    
    @dispatch(dict)
    def __init__(self, cfg: dict):
        super().__init__()
        self.model = utils.get_obj(cfg.model)(**cfg.model_params)
        self.optimizer = utils.get_obj(cfg.optimizer)(self.model.parameters(), **cfg.optimizer_params) if 'optimizer' in cfg else None
        self.scheduler = utils.get_obj(cfg.scheduler)(self.optimizer, **cfg.scheduler_params) if 'scheduler' in cfg else None
        self.criterion = utils.get_obj(cfg.criterion)(**cfg.criterion_params) if 'criterion' in cfg else None
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def calc_loss(self, batch):
        pass

    def training_step(self, batch, batch_idx: int):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch)
        self.log('test_loss', loss, prog_bar=True)
    
    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler


class BirdCLEFModule(BaseModule):
    def forward(self, batch):
        return self.model(batch)

    def calc_loss(self, batch):
        return self.criterion(
            batch['output']['logits'], 
            batch['target'], 
            batch['weight']
        )
        