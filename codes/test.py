import torch
import utils
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer


def test(cfg: dict):
    tester = Trainer(
        logger=False,
        **cfg.trainer_params,
    )

    if cfg.checkpoint_path.endswith('.ckpt'):
        tester.test(
            utils.get_instance(cfg.lightning_module, cfg.lightning_module_params),
            datamodule=DataModule(cfg.mode, **cfg.datamodule_params),
            ckpt_path=cfg.checkpoint_path
        )
    elif cfg.checkpoint_path.endswith('_pickle.pt'):
        tester.test(
            torch.load(cfg.checkpoint_path),
            datamodule=DataModule(cfg.mode, **cfg.datamodule_params),
        )
    else:
        raise ValueError(f'Invalid checkpoint path: {cfg.checkpoint_path}')