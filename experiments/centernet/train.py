import torch
import numpy as np
import matplotlib.pyplot as plt
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path

from typing import Any
from torch import Tensor
from logging import getLogger, FileHandler
from sklearn.model_selection import StratifiedKFold
from app.dataset.wheat import WheatDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from object_detection.metrics import MeanPrecition
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Visualize,
    Trainer as _Trainer,
    Criterion,
    ToBoxes,
)
from object_detection.models.mkmaps import MkCenterBoxMaps
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.models.losses import DIoU
from object_detection import boxmap_to_boxes, yolo_to_pascal, BoxMap, YoloBoxes
from app.preprocess import kfold
from experiments.centernet import config
from app.transforms import get_train_transforms, get_valid_transforms


class Trainer(_Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )

    def train_one_epoch(self) -> None:
        super().train_one_epoch()
        self.lr_scheduler.step()


def train(epochs: int) -> None:
    train_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        transforms=get_train_transforms(),
    )
    test_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        transforms=get_valid_transforms(),
    )
    fold_keys = [x[2].shape[0] // 30 for x in test_dataset.rows]
    train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[
        config.fold_idx
    ]

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=config.batch_size * 2,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    backbone = EfficientNetBackbone(
        config.effdet_id, out_channels=config.channels, pretrained=config.pretrained
    )
    model = CenterNet(
        channels=config.channels,
        backbone=backbone,
        out_idx=config.out_idx,
        depth=config.depth,
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion(
        heatmap_weight=config.heatmap_weight,
        box_weight=config.box_weight,
        mk_hmmaps=config.mkmaps,
        mk_boxmaps=MkCenterBoxMaps(),
    )

    visualize = Visualize(config.out_dir, "centernet", limit=5, show_probs=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_loader=model_loader,
        optimizer=optimizer,
        visualize=visualize,
        device=config.device,
        criterion=criterion,
        get_score=MeanPrecition(),
        to_boxes=config.to_boxes,
    )(epochs)


if __name__ == "__main__":
    train(1000)
