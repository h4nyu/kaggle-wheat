from typing import List, Tuple
from typing_extensions import Literal
from object_detection.entities import PyramidIdx
from object_detection.models.centernetv1 import ( 
    MkCrossMaps
)
from object_detection.models.backbones.effnet import Phi

import torch

test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"

n_splits = 9
fold_idx = 0
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
metric: Tuple[str, Literal["max", "min"]] = ("test_loss", "min")
max_size = 512
batch_size = 16
num_workers = 8

# model
effdet_id: Phi = 3
out_idx: PyramidIdx = 4
fpn_depth = 1
hm_depth = 2
box_depth = 2
channels = 64
pretrained = True
anchor_size=1

# heatmap
heatmap_weight = 1.0
box_weight = 40.0

# ToBoxes
confidence_threshold = 0.40
use_peak = True

# box merge
iou_threshold = 0.6
final_threshold = 0.0

mkmaps = MkCrossMaps()

out_dir = f"/kaggle/input/models/ctdtv1-effdet_id-{effdet_id}-fpn_depth-{fpn_depth}-hm_depth-{hm_depth}-box_depth-{box_depth}-channels-{channels}-out_idx-{out_idx}-max_size-{max_size}/{fold_idx}"
