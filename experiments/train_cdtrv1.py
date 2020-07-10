import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from app.dataset.wheat import WheatDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from object_detection.meters import BestWatcher
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
)
from object_detection.entities import PyramidIdx
from object_detection.model_loader import ModelLoader
from app import config
from app.preprocess import kfold

### config ###
fold_idx = 0
channels = 128
depth = 1
lr = 1e-4
max_size = 512
batch_size = 12
out_idx: PyramidIdx = 5
box_threshold = 0.2
sigma = 6.0
heatmap_weight = 1.0
sizemap_weight = 2.0
to_boxes_kernel_size = 5

box_limit = 100
out_dir = f"/kaggle/input/models/ctdtv1/{fold_idx}"
### config ###

train_dataset = WheatDataset(
    image_dir=config.train_image_dir, annot_file=config.annot_file, max_size=max_size, mode="train",
)
test_dataset = WheatDataset(
    image_dir=config.train_image_dir, annot_file=config.annot_file, max_size=max_size, mode="test"
)
fold_keys = [x[2].shape[0] // 30 for x in test_dataset.rows]
train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[fold_idx]

train_loader = DataLoader(
    Subset(train_dataset, train_idx),
    batch_size=batch_size,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    Subset(test_dataset, test_idx),
    batch_size=batch_size,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
backbone = EfficientNetBackbone(3, out_channels=channels)
model = CenterNetV1(channels=channels, backbone=backbone, out_idx=out_idx, depth=depth)
model_loader = ModelLoader(out_dir=out_dir)
criterion = Criterion(
    heatmap_weight=heatmap_weight, sizemap_weight=sizemap_weight, sigma=sigma
)

visualize = Visualize(out_dir, "centernet", limit=10, show_probs=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,)
best_watcher = BestWatcher(mode="max")
to_boxes = ToBoxes(threshold=box_threshold, limit=box_limit, kernel_size=to_boxes_kernel_size)
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
    best_watcher=best_watcher,
    to_boxes=to_boxes,
)
trainer.train(1000)
