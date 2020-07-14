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
from object_detection.models.centernetv1 import (
    collate_fn,
    CenterNetV1,
    Visualize,
    Trainer,
    Criterion,
    ToBoxes,
    Anchors,
    BoxMerge,
)
from object_detection.model_loader import ModelLoader ,BestWatcher
from app.preprocess import kfold
from . import config

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
train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[config.fold_idx]

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
backbone = EfficientNetBackbone(3, out_channels=config.channels)
model = CenterNetV1(
    channels=config.channels,
    backbone=backbone,
    out_idx=config.out_idx,
    fpn_depth=config.fpn_depth,
    hm_depth=config.hm_depth,
    box_depth=config.box_depth,
)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key="test_hm",
    best_watcher = BestWatcher(mode="min")
)
criterion = Criterion(
    heatmap_weight=config.heatmap_weight, box_weight=config.box_weight, sigma=config.sigma,
)

visualize = Visualize(config.out_dir, "centernet", limit=10, show_probs=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,)
to_boxes = ToBoxes(threshold=config.confidence_threshold, use_peak=config.use_peak,)
box_merge = BoxMerge(iou_threshold=config.iou_threshold)
trainer = Trainer(
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
)
