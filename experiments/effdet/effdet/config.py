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
batch_size = 12
num_workers = 8

## model
effdet_id: Phi = 3
out_ids: List[PyramidIdx] = [5, 6, 7]
channels = 128
pretrained = True

## criterion
label_weight = 2.0
pos_threshold = 0.4
size_threshold = 0.4
label_thresholds = (0.3, 0.4)

## anchor
anchor_ratios = [1.0]
anchor_scales = [2.0]
anchor_size = 1

## ToBoxes
confidence_threshold = 0.42

## BoxMerge
iou_threshold = 0.50
final_threshold = 0.0

out_dir = f"/kaggle/input/models/effdet-effdet_id-{effdet_id}-anchor_size-{anchor_size}-out_ids-{len(out_ids)}-channels-{channels}-max_size-{max_size}/{fold_idx}"
