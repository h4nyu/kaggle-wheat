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
from object_detection.models.centernetv1 import (
    collate_fn,
    CenterNetV1,
    Visualize,
    Trainer as _Trainer,
    Criterion,
    ToBoxes,
    Anchors,
    Heatmap,
)
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.models.losses import DIoU
from object_detection import boxmap_to_boxes, yolo_to_pascal, BoxMap, YoloBoxes
from app.preprocess import kfold
from . import config


# class BoxLoss:
#     def __init__(self, threshold: float = 0.5) -> None:
#         self.diou = DIoU()
#         self.threshold = threshold

#     def __call__(
#         self, anchormap: BoxMap, diffmap: BoxMap, gt_boxes: YoloBoxes, heatmap: Heatmap,
#     ) -> Tensor:
#         device = diffmap.device
#         box_diffs = boxmap_to_boxes(diffmap)
#         anchors = boxmap_to_boxes(anchormap)
#         pred_boxes = YoloBoxes(anchors + box_diffs)
#         loss_matrix = self.diou(
#             yolo_to_pascal(pred_boxes, (1, 1),), yolo_to_pascal(gt_boxes, (1, 1),)
#         )
#         loss_min, match_indices = torch.min(loss_matrix, dim=1)
#         positive_indices = loss_min < self.threshold
#         if positive_indices.sum() == 0:
#             return torch.tensor(0.0).to(device)
#         loss = loss_matrix[positive_indices].mean()
#         return loss


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
    model = CenterNetV1(
        channels=config.channels,
        backbone=backbone,
        out_idx=config.out_idx,
        fpn_depth=config.fpn_depth,
        hm_depth=config.hm_depth,
        box_depth=config.box_depth,
        anchors=Anchors(size=config.anchor_size),
    )
    model_loader = ModelLoader(
        out_dir=config.out_dir,
        key=config.metric[0],
        best_watcher=BestWatcher(mode=config.metric[1]),
    )
    criterion = Criterion(
        heatmap_weight=config.heatmap_weight,
        box_weight=config.box_weight,
        mkmaps=config.mkmaps,
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
