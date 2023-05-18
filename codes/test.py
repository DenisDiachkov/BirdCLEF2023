import torch
import utils
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset.datamodule import DataModule


def get_new_test_func(test_func):
    def new_test_func(x, batch_idx):
        output = test_func(x, batch_idx)
    return new_test_func


def test(cfg: dict):
    tester = Trainer(
        logger=False,
        **cfg.trainer_params,
    )

    if cfg.checkpoint_path.endswith('.ckpt'):
        module = utils.get_instance(cfg.lightning_module, cfg.lightning_module_params)
        module = module.load_from_checkpoint(cfg.checkpoint_path)
    elif cfg.checkpoint_path.endswith('_pickle.pt'):
        module = torch.load(cfg.checkpoint_path)
    module.test_step = get_new_test_func(module.test_step)
    datamodule=DataModule(cfg.mode, **cfg.datamodule_params)
    tester.test(
        module,
        datamodule=datamodule
    )