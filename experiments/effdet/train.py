import torch
import numpy as np
import typing as t
import matplotlib.pyplot as plt
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path

from logging import getLogger, FileHandler
from sklearn.model_selection import StratifiedKFold
from app.dataset.wheat import WheatDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from object_detection.metrics import MeanPrecition
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.effidet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
    BoxLoss,
    LabelLoss,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app.preprocess import kfold
from experiments.effdet import config
from app.transforms import get_train_transforms, get_valid_transforms


def train(epochs: int) -> None:
    train_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        transforms=get_train_transforms(config.max_size),
    )
    test_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        transforms=get_valid_transforms(config.max_size),
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
    anchors = Anchors(
            size=config.anchor_size,
            ratios=config.anchor_ratios,
            scales=config.anchor_scales,
    )
    model = EfficientDet(
        num_classes=1,
        channels=config.channels,
        backbone=backbone,
        anchors=anchors,
        out_ids=config.out_ids,
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion()
    visualize = Visualize(config.out_dir, "test", limit=5, show_probs=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    to_boxes = ToBoxes(
        confidence_threshold=config.confidence_threshold
    )
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
        to_boxes=to_boxes,
    )(epochs)


if __name__ == "__main__":
    train(1000)
