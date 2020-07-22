import torch
import numpy as np
from . import config
from typing import List, Tuple, Any
from torch.utils.data import DataLoader, Subset
from object_detection.models.backbones.effnet import EfficientNetBackbone
from app.dataset.wheat import WheatDataset
from object_detection.entities import TrainSample, ImageBatch, ImageId
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.metrics import MeanPrecition
from object_detection.models.box_merge import BoxMerge
from object_detection.models.efficientdet import (
    EfficientDet,
    Predictor,
    ToBoxes,
    Anchors,
)


def _collate_fn(batch: List[TrainSample],) -> Tuple[ImageBatch, List[ImageId]]:
    images: List[Any] = []
    id_batch: List[ImageId] = []
    for id, img, _, _ in batch:
        images.append(img)
        id_batch.append(id)
    return ImageBatch(torch.stack(images)), id_batch


def evaluate(limit: int = 100) -> None:
    backbone = EfficientNetBackbone(config.effdet_id, out_channels=config.channels)
    anchors = Anchors(
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
    box_merge = BoxMerge(
        iou_threshold=config.iou_threshold, confidence_threshold=config.final_threshold
    )
    dataset = Subset(
        WheatDataset(
            annot_file=config.annot_file,
            image_dir=config.train_image_dir,
            max_size=config.max_size,
            mode="test",
        ),
        list(range(limit)),
    )
    to_boxes = ToBoxes(confidence_threshold=config.confidence_threshold)
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=_collate_fn,
        batch_size=config.batch_size * 2,
        shuffle=False,
    )
    predictor = Predictor(
        model=model,
        loader=data_loader,
        model_loader=model_loader,
        device=config.device,
        box_merge=box_merge,
        to_boxes=to_boxes,
    )
    boxes_list, confs_list, ids = predictor()
    gt_boxes_list = [dataset[i][2] for i in range(len(dataset))]
    get_score = MeanPrecition()
    score = np.mean(
        [get_score(x, y.to(x.device)) for x, y in zip(boxes_list, gt_boxes_list)]
    )
    print(score)
