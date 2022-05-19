"""
comet/torch/lightning.py
"""
from __future__ import absolute_import, division, print_function, annotations

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import (
    CometLogger,
    # MLFlowLogger,
    # NeptuneLogger,
    # TensorBoardLogger,
    WandbLogger,
)

from omegaconf import DictConfig
import hydra

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
import pytorch_lightning as pl
from dataclasses import dataclass


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


def setup_wandb():
    return WandbLogger(project='mlops')


def setup_CometLogger(cfg: DictConfig) -> None:
    return CometLogger(
        save_dir='.',
        workspace=cfg.get('workspace'),                         # optional
        api_key=cfg.comet.get('api_key'),
        project_name=cfg.comet.get('project_name'),             # optional
        rest_api_key=cfg.comet.get('rest_api_key'),             # optional
        experiment_key=cfg.comet.get('experiment_key'),         # optional
        experiment_name=cfg.comet.get('experiment_name'),       # optional
    )


class LitMNIST(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
        )
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        hidden_size = self.cfg.get('hidden_size', 10)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def prepare_data(self):
        MNIST(self.cfg.data_dir, train=True, download=True)
        MNIST(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train / val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(
                self.cfg.data_dir,
                train=True,
                transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                self.cfg.data_dir,
                train=False,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

    def forward(self, x):
        # in lightning, forward defines the prediction / inference actions
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def validation_step(self, batch, batch_idx):  # type:ignore
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # Training step defines the train loop. It is independent of forward
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    logger = None
    if cfg.get('comet', None) is not None:
        logger = setup_CometLogger(cfg)

    model = LitMNIST(cfg)
    trainer = pl.Trainer(
        logger=[logger],
        gpus=AVAIL_GPUS,
        max_epochs=cfg.epochs,
    )
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
