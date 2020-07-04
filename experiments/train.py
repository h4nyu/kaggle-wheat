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
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Trainer,
    Visualize,
    Criterion,
    Reg,
    ToBoxes,
)
from object_detection.entities import PyramidIdx
from object_detection.model_loader import ModelLoader
from app import config
from app.preprocess import kfold

## config
fold_idx = 0
channels = 128
lr = 1e-3
max_size = 512
batch_size = 6
out_idx: PyramidIdx = 5
###

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
model = CenterNet(channels=channels, backbone=backbone, out_idx=out_idx)
model_loader = ModelLoader(out_dir=out_dir)
criterion = Criterion(heatmap_weight=5.0, sigma=0.3)

visualize = Visualize(out_dir, "centernet", limit=10, show_probs=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,)
best_watcher = BestWatcher(mode="max")
to_boxes = ToBoxes(threshold=0.3, limit=100)
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
