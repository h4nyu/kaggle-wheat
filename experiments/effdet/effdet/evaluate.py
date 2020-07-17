import torch
import numpy as np
from . import config
from typing import List, Tuple, Any
from torch.utils.data import DataLoader, Subset
from object_detection.models.backbones.effnet import EfficientNetBackbone
from object_detection.models.centernetv1 import (
    Predictor,
)
from app.dataset.wheat import WheatDataset
from object_detection.entities import TrainSample, ImageBatch, ImageId
from object_detection.model_loader import ModelLoader, BestWatcher
from object_detection.metrics import MeanPrecition
from object_detection.models.centernetv1 import (
    CenterNetV1,
    ToBoxes,
)


def _collate_fn(batch: List[TrainSample],) -> Tuple[ImageBatch, List[ImageId]]:
    images: List[Any] = []
    id_batch: List[ImageId] = []
    for id, img, _, _ in batch:
        images.append(img)
        id_batch.append(id)
    return ImageBatch(torch.stack(images)), id_batch


def evaluate(limit:int=100) -> None:
    ...
