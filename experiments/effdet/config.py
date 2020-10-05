from typing import List, Tuple
from typing_extensions import Literal
from object_detection.entities import PyramidIdx
from object_detection.models.backbones.effnet import Phi
import torch

test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"

n_splits = 5
fold_idx = 0
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
metric: Tuple[str, Literal["max", "min"]] = ("score", "max")
max_size = 512
batch_size = 8
num_workers = 8

## model
effdet_id: Phi = 4
out_ids: List[PyramidIdx] = [4, 5, 6]
channels = 128
pretrained = True

## criterion
cls_weight = 1.0
box_weight = 10.0

## anchor
anchor_ratios = [1.0]
anchor_scales = [1.0]
anchor_size = 4.0

## ToBoxes
confidence_threshold = 0.3

out_dir = f"/kaggle/input/models/2020-10-03"
