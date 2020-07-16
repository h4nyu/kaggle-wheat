import torch
from app.dataset.wheat import WheatDataset, PredictionDataset
from torch.utils.data import DataLoader
from app import config
from pathlib import Path
from object_detection.utils import DetectionPlot
from app import config


def test_train_dataset() -> None:
    dataset = WheatDataset(config.annot_file, config.train_image_dir, max_size=1024)
    image_id, img, boxes, _ = dataset[0]
    assert img.dtype == torch.float32
    assert boxes.dtype == torch.float32

    for i in range(10):
        _, img, boxes, _ = dataset[100]
        _, h, w = img.shape
        plot = DetectionPlot(figsize=(20, 20), w=w, h=h)
        plot.with_image(img)
        plot.with_yolo_boxes(boxes, color="red")
        plot.save(f"{config.working_dir}/test-dataset-{i}.png")


def test_prediction_dataset() -> None:
    dataset = PredictionDataset(
        "/kaggle/input/global-wheat-detection/test", max_size=512
    )
    img_id, img = dataset[0]
    assert img.dtype == torch.float32
