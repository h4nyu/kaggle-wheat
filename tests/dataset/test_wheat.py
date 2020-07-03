import torch
from app.dataset.wheat import WheatDataset
from torch.utils.data import DataLoader
from app import config
from pathlib import Path
from object_detection.utils import DetectionPlot
from app import config


def test_plotrow() -> None:
    dataset = WheatDataset(config.annot_file, config.train_image_dir)
    image_id, img, boxes, _ = dataset[0]
    assert img.dtype == torch.float32
    assert boxes.dtype == torch.float32

    for i in range(5):
        _, img, boxes, _ = dataset[i]
        _, h, w = img.shape
        plot = DetectionPlot(figsize=(10, 10), w=w, h=h)
        plot.with_image(img)
        plot.with_yolo_boxes(boxes, color="red")
        plot.save(f"{config.working_dir}/test-dataset-{i}.png")
