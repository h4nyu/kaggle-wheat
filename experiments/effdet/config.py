from typing import List, Tuple
from typing_extensions import Literal
from object_detection.entities import PyramidIdx
from object_detection.models.backbones.effnet import Phi
import torch

test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"

n_splits = 10
fold_idx = 0
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
metric: Tuple[str, Literal["max", "min"]] = ("test_loss", "min")
max_size = 512
batch_size = 5
num_workers = 4

## model
effdet_id: Phi = 1
out_ids: List[PyramidIdx] = [3, 4, 5, 6]
channels = 64
pretrained = True

## criterion
label_weight = 2.0
pos_threshold = 0.4
size_threshold = 0.4
label_thresholds = (0.4, 0.5)

## anchor
anchor_ratios = [1.0, 2/3, 3/2]
anchor_scales = [1.0]
anchor_size = 3.0

## ToBoxes
confidence_threshold = 0.3

out_dir = f"/kaggle/input/models/2020-09-02-0"
