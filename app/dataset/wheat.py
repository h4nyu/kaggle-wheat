import typing as t
import numpy as np
import pandas as pd
import torch
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
    image_path = f"{image_dir}/{image_id}.jpg"
    return imread(image_path)


class WheatDataset(Dataset):
    def __init__(
        self,
        annot_file: str,
        image_dir: str,
        max_size: int = 512,
        mode: Literal["train", "test"] = "train",
    ) -> None:
        super().__init__()
        self.annot_file = Path(annot_file)
        self.rows = load_lables(annot_file)
        self.mode = mode
        self.image_dir = Path(image_dir)
        self.cache: t.Dict[str, t.Any] = dict()
        self.image_dir = Path(image_dir)

        bbox_params = {"format": "coco", "label_fields": ["labels"]}
        self.pre_transforms = A.Compose(
            [A.LongestMaxSize(max_size=max_size),], bbox_params=bbox_params,
        )
        self.train_transforms = A.Compose(
            [
                A.RandomResizedCrop(
                    p=1, height=max_size, width=max_size, scale=(0.9, 1.1)
                ),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                #  A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=bbox_params,
        )

        self.post_transforms = A.Compose([ToTensorV2(),])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainSample:
        image_id, _, boxes = self.rows[index]
        image = get_img(image_id=image_id, image_dir=self.image_dir)
        labels = np.zeros(boxes.shape[0])
        res = self.pre_transforms(image=image, bboxes=boxes, labels=labels)
        if self.mode == "train":
            res = self.train_transforms(**res)
        res = self.post_transforms(**res)

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
