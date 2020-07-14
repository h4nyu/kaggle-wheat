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
from object_detection.entities import PyramidIdx
from object_detection.model_loader import ModelLoader ,BestWatcher
from app import config
from app.preprocess import kfold

### config ###
fold_idx = 0
lr = 1e-4
max_size = 512
batch_size = 16

depth = 1
out_idx: PyramidIdx = 5
channels = 64

sigma = 1.0
heatmap_weight = 1.0
box_weight = 20.0
iou_threshold = 0.6
use_peak = False

confidence_threshold = 0.3

fpn_depth = 1
hm_depth = 1
box_depth = 1

out_dir = f"/kaggle/input/models/ctdtv1-fpn_depth-{fpn_depth}-hm_depth-{hm_depth}-box_depth-{box_depth}-channels-{channels}-out_idx-{out_idx}-max_size-{max_size}/{fold_idx}"

### config ###

Path(out_dir).mkdir(exist_ok=True, parents=True)
logger = getLogger()
file_handler = FileHandler(filename=f"{out_dir}/train.log")
logger.addHandler(file_handler)
train_dataset = WheatDataset(
    image_dir=config.train_image_dir,
    annot_file=config.annot_file,
    max_size=max_size,
    mode="train",
)
test_dataset = WheatDataset(
    image_dir=config.train_image_dir,
    annot_file=config.annot_file,
    max_size=max_size,
    mode="test",
)
fold_keys = [x[2].shape[0] // 30 for x in test_dataset.rows]
train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[fold_idx]

train_loader = DataLoader(
    Subset(train_dataset, train_idx),
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    Subset(test_dataset, test_idx),
    batch_size=batch_size,
    drop_last=False,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
backbone = EfficientNetBackbone(3, out_channels=channels)
model = CenterNetV1(
    channels=channels,
    backbone=backbone,
    out_idx=out_idx,
    fpn_depth=fpn_depth,
    hm_depth=hm_depth,
    box_depth=box_depth,
)
model_loader = ModelLoader(
    out_dir=out_dir,
    key="test_hm",
    best_watcher = BestWatcher(mode="min")
)
criterion = Criterion(
    heatmap_weight=heatmap_weight, box_weight=box_weight, sigma=sigma,
)

visualize = Visualize(out_dir, "centernet", limit=10, show_probs=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,)
to_boxes = ToBoxes(threshold=confidence_threshold, use_peak=use_peak,)
box_merge = BoxMerge(iou_threshold=iou_threshold)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    device="cuda",
    criterion=criterion,
    get_score=MeanPrecition(),
    to_boxes=to_boxes,
    box_merge=box_merge,
)
trainer(1000)
