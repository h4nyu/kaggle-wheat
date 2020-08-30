from typing import List, Tuple
from typing_extensions import Literal
from object_detection.entities import PyramidIdx
from object_detection.models.centernetv1 import (
    MkCrossMaps,
    MkGaussianMaps,
    ToBoxes,
)
from object_detection.models.backbones.effnet import Phi

import torch

id = "2020-08-29-0"
test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"

n_splits = 5
fold_idx = 0
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
metric: Tuple[str, Literal["max", "min"]] = ("score", "max")
max_size = 512 * 2
batch_size = 3
num_workers = 8

# lr_scheduler
T_max = 20
eta_min = 1e-6

# model
effdet_id: Phi = 3
out_idx: PyramidIdx = 4
fpn_depth = 1
hm_depth = 1
box_depth = 1
channels = 64
pretrained = True
anchor_size = 1

# heatmap
heatmap_weight = 1.0
box_weight = 20.0

# ToBoxes
confidence_threshold = 0.1
use_peak = True

# box merge
iou_threshold = 0.6
final_threshold = confidence_threshold

mkmaps = MkGaussianMaps(sigma=0.5, mode="constant")

to_boxes = ToBoxes(threshold=confidence_threshold, use_peak=use_peak, kernel_size=3)

out_dir = f"/kaggle/input/models/{id}/{fold_idx}"
