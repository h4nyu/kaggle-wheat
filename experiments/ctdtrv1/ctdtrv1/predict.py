from object_detection.models.centernetv1 import Predictor, prediction_collate_fn
from torch.utils.data import DataLoader

from app.dataset.wheat import PredictionDataset
from . import config as cfg
from . import train

dataset = PredictionDataset(
    image_dir=cfg.test_image_dir,
    max_size=cfg.max_size,
)

data_loader = DataLoader(
    dataset=dataset,
    collate_fn=prediction_collate_fn,
    batch_size=cfg.batch_size,
    shuffle=True,
)

predictor = Predictor(
    model=train.model,
    loader=data_loader,
    model_loader=train.model_loader,
    device="cuda",
    box_merge=train.box_merge,
    to_boxes=train.to_boxes,
)
