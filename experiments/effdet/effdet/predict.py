from typing import List, Tuple
from object_detection.entities import YoloBoxes, Confidences, ImageId
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


def predict() -> Tuple[List[YoloBoxes], List[Confidences], List[ImageId]]:
    ...
