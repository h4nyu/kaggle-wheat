from object_detection.models.centernetv1 import (
    Predictor,
    prediction_collate_fn,
    BoxMerge,
    CenterNetV1,
    ToBoxes,
)
from torch.utils.data import DataLoader
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.model_loader import ModelLoader, BestWatcher

from app.dataset.wheat import PredictionDataset
from . import config


def predict() -> None:
    backbone = EfficientNetBackbone(3, out_channels=config.channels)
    model = CenterNetV1(
        channels=config.channels,
        backbone=backbone,
        out_idx=config.out_idx,
        fpn_depth=config.fpn_depth,
        hm_depth=config.hm_depth,
        box_depth=config.box_depth,
    )
    dataset = PredictionDataset(
        image_dir=config.test_image_dir, max_size=config.max_size,
    )

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=prediction_collate_fn,
        batch_size=config.batch_size,
        shuffle=True,
    )
    box_merge = BoxMerge(iou_threshold=config.iou_threshold)
    model_loader = ModelLoader(
        out_dir=config.out_dir, key="test_hm", best_watcher=BestWatcher(mode="min")
    )
    to_boxes = ToBoxes(threshold=config.confidence_threshold, use_peak=config.use_peak,)

    predictor = Predictor(
        model=model,
        loader=data_loader,
        model_loader=model_loader,
        device=config.device,
        box_merge=box_merge,
        to_boxes=to_boxes,
    )
