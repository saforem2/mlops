"""
comet/torch/lightning_optimizer.py """
from __future__ import absolute_import, division, print_function, annotations
import os
import sys
from omegaconf import DictConfig

import hydra
from omegaconf import OmegaConf, DictConfig
from argparse import Namespace


here = os.path.abspath(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.append(here)

# from .lightning import LitMNIST, setup_CometLogger


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    from comet_ml import Optimizer
    from comet_ml.config import get_config

    from lightning import LitMNIST, setup_CometLogger
    import pytorch_lightning as pl

    optimizer = None
    if cfg.lightning is not None:
        optimizer = Optimizer(config=OmegaConf.to_container(cfg.lightning))

    if optimizer is not None:
        for parameters in optimizer.get_parameters():
            cfg = OmegaConf.create(parameters)
            model = LitMNIST(cfg)
            comet_logger = setup_CometLogger(cfg)
            trainer = pl.Trainer(
                max_epochs=1,
                # early_stop_callback=True,  # requires val_loss be logged
                logger=[comet_logger],
                # num_processes=2,
                # distributed_backend='ddp_cpu',
            )

            trainer.fit(model)


if __name__ == '__main__':
    main()
