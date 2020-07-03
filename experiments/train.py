import torch
import numpy as np
from cytoolz.curried import groupby, valmap, pipe, unique, map, reduce
from pathlib import Path
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from app.dataset.wheat import WheatDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from object_detection.models.backbones import ResNetBackbone
from object_detection.models.centernet import (
    collate_fn,
    CenterNet,
    Trainer,
    Visualize,
    Criterion,
    Reg,
)
from object_detection.model_loader import ModelLoader
from app import config
from app.preprocess import kfold

## config
fold_idx = 0
channels = 128
###

dataset = WheatDataset(
    image_dir=config.train_image_dir,
    annot_file=config.annot_file,
    max_size=config.max_size,
)
fold_keys = [x[2].shape[0] // 20 for x in dataset.rows]
train_idx, test_idx = list(kfold(n_splits=config.n_splits, keys=fold_keys))[fold_idx]

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=config.batch_size,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=config.batch_size,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)
backbone = ResNetBackbone("resnet34", out_channels=channels)
out_dir = f"/kaggle/input/models/{fold_idx}"
model = CenterNet(channels=channels, backbone=backbone, out_idx=4)
model_loader = ModelLoader(out_dir=out_dir)
criterion = Criterion(sizemap_weight=1.0, sigma=0.3)

visualize = Visualize("./", "centernet", limit=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,)
trainer = Trainer(
    model,
    train_loader,
    test_loader,
    model_loader,
    optimizer,
    Visualize("/store/centernet", "test", limit=2),
    "cuda",
    criterion=criterion,
)
trainer.train(100)
