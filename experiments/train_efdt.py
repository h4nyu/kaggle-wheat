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
from object_detection.models.backbones.resnet import ResNetBackbone
from object_detection.models.efficientdet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
)
from object_detection.entities import PyramidIdx
from object_detection.model_loader import ModelLoader
from app import config
from app.preprocess import kfold

### config ###
fold_idx = 0
channels = 128
depth = 3
lr = 1e-3
max_size = 512
batch_size = 7
out_idx: PyramidIdx = 4
box_threshold = 0.1
sigma = 3.0
confidence_threshold = 0.5
nms_threshold = 0.3
channels = 128

box_limit = 100
### config ###

dataset = WheatDataset(
    image_dir=config.train_image_dir, annot_file=config.annot_file, max_size=max_size,
)
fold_keys = [x[2].shape[0] // 20 for x in dataset.rows]
train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[fold_idx]

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=batch_size,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=batch_size,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
backbone = ResNetBackbone("resnet50", out_channels=channels)
out_dir = f"/kaggle/input/models/{fold_idx}"
model = EfficientDet(num_classes=1, channels=channels, backbone=backbone)
model_loader = ModelLoader(out_dir=out_dir)
criterion = Criterion()
visualize = Visualize(out_dir, "centernet", limit=10, show_probs=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,)
best_watcher = BestWatcher(mode="max")
get_score = MeanPrecition()
to_boxes = ToBoxes(
    nms_threshold=nms_threshold, confidence_threshold=confidence_threshold,
)

trainer = Trainer(
    model,
    train_loader=train_loader,
    test_loader=test_loader,
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    criterion=criterion,
    best_watcher=best_watcher,
    get_score=get_score,
    device="cuda",
    to_boxes=to_boxes,
)
trainer.train(1000)
