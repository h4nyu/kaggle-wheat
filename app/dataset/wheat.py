import typing as t
import numpy as np
import pandas as pd
import torch
import cv2
import torchvision
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
from glob import glob
from object_detection.entities import (
    TrainSample,
    PredictionSample,
    ImageSize,
    ImageId,
    Image,
    CoCoBoxes,
    coco_to_yolo,
    Labels,
)
from PIL import Image as PILImage
from albumentations.pytorch.transforms import ToTensorV2
from skimage.io import imread
from cytoolz.curried import pipe, groupby, valmap
from typing_extensions import Literal
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

Row = t.Tuple[Tensor, Tensor, str]
Batch = t.Sequence[Row]


def parse_boxes(strs: t.List[str]) -> CoCoBoxes:
    coco = CoCoBoxes(
        torch.tensor(np.stack([np.fromstring(s[1:-1], sep=",") for s in strs]))
    )
    return coco


def load_lables(
    annot_file: str, limit: t.Optional[int] = None
) -> t.Sequence[t.Tuple[ImageId, ImageSize, CoCoBoxes]]:
    df = pd.read_csv(annot_file, nrows=limit)
    df["bbox"] = df["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
    return df.groupby("image_id").apply(
        lambda x: (
            ImageId(x["image_id"].iloc[0]),
            (x["width"].iloc[0], x["height"].iloc[0]),
            (torch.tensor(np.stack(x["bbox"]))),
        )
    )


def get_img(image_id: ImageId, image_dir: Path) -> t.Any:
    return np.array(PILImage.open(f"{image_dir}/{image_id}.jpg"))

DEFAULT_TRANSFORM = A.Compose(
    [A.Resize(height=1024, width=1024, p=1.0), ToTensorV2(p=1.0),],
    p=1.0,
    bbox_params=A.BboxParams(
        format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
    ),
)


class WheatDataset(Dataset):
    def __init__(
        self,
        annot_file: str,
        image_dir: str,
        transforms: t.Callable = DEFAULT_TRANSFORM,
    ) -> None:
        super().__init__()
        self.annot_file = Path(annot_file)
        self.rows = load_lables(annot_file)
        self.image_dir = Path(image_dir)
        self.cache: t.Dict[str, t.Any] = dict()
        self.image_dir = Path(image_dir)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainSample:
        image_id, _, boxes = self.rows[index]
        image = get_img(image_id=image_id, image_dir=self.image_dir)
        labels = np.zeros(boxes.shape[0])
        res = self.transforms(image=image, bboxes=boxes, labels=labels)
        image = Image(res["image"].float() / 255.0)
        _, h, w = image.shape
        boxes = CoCoBoxes(torch.tensor(res["bboxes"]).float())
        yolo_boxes = coco_to_yolo(boxes, (w, h))
        return (image_id, image, yolo_boxes, Labels(torch.from_numpy(labels)))


class PredictionDataset(Dataset):
    def __init__(self, image_dir: str, max_size: int) -> None:
        rows: t.List[t.Tuple[ImageId, Path]] = []
        for p in glob(f"{image_dir}/*.jpg"):
            path = Path(p)
            rows.append((ImageId(path.stem), path))
        self.rows = rows
        self.transforms = A.Compose(
            [A.LongestMaxSize(max_size=max_size), ToTensorV2(),]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PredictionSample:
        image_id, path = self.rows[index]
        img = imread(path)
        img = self.transforms(image=img)["image"] / 255.0
        return image_id, Image(img)
