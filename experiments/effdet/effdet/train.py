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
from object_detection.models.box_merge import BoxMerge
from object_detection.models.efficientdet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
    SizeLoss,
    PosLoss,
    LabelLoss,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from app.preprocess import kfold
from . import config


def train(epochs: int) -> None:
    train_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        max_size=config.max_size,
        mode="train",
    )
    test_dataset = WheatDataset(
        image_dir=config.train_image_dir,
        annot_file=config.annot_file,
        max_size=config.max_size,
        mode="test",
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
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    backbone = EfficientNetBackbone(
        config.effdet_id, out_channels=config.channels, pretrained=config.pretrained
    )
    anchors = Anchors(ratios=config.anchor_ratios, scales=config.anchor_scales,)
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
    box_merge = BoxMerge(
        iou_threshold=config.iou_threshold, confidence_threshold=config.final_threshold
    )
    criterion = Criterion(
        label_weight=config.label_weight,
        pos_loss=PosLoss(iou_threshold=config.pos_threshold),
        size_loss=SizeLoss(iou_threshold=config.size_threshold),
        label_loss=LabelLoss(iou_thresholds=config.label_thresholds),
    )
    visualize = Visualize(config.out_dir, "test", limit=5, show_probs=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,)
    to_boxes = ToBoxes(confidence_threshold=config.confidence_threshold)
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
        box_merge=box_merge,
    )(epochs)
