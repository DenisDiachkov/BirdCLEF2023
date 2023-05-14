import torch
import utils
import pytorch_lightning as pl
from pytorch_lightning import Trainer


def get_new_test_func(test_func):
    def new_test_func(x, batch_idx):
        output = test_func(x, batch_idx)
        print(x['input'].shape)
    return new_test_func


def test(cfg: dict):
    tester = Trainer(
        logger=False,
        **cfg.trainer_params,
    )

    if cfg.checkpoint_path.endswith('.ckpt'):
        tester.test(
            utils.get_instance(cfg.lightning_module, cfg.lightning_module_params),
            datamodule=pl.DataModule(cfg.mode, **cfg.datamodule_params),
            ckpt_path=cfg.checkpoint_path
        )
    elif cfg.checkpoint_path.endswith('_pickle.pt'):
        module = torch.load(cfg.checkpoint_path)
        datamodule = torch.load(cfg.datamodule_path)
        if 'dataset_params' in cfg:
            datamodule.setup_test_data(cfg.dataset_params) 
        module.test_step = get_new_test_func(module.test_step)
        tester.test(
            module,
            datamodule=datamodule,
        )
    else:
        raise ValueError(f'Invalid checkpoint path: {cfg.checkpoint_path}')