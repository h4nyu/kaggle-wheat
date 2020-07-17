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
lr = 5e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
metric:Tuple[str, Literal['max', 'min']] = ("score", "max")
max_size = 512
batch_size = 10
num_workers = 8

## model
effdet_id:Phi = 4
out_ids: List[PyramidIdx] = [5,6]
channels = 128

## anchor
anchor_ratios = [0.33, 1, 3]
anchor_size = 2

## ToBoxes
iou_threshold = 0.50

confidence_threshold = 0.32
final_threshold = 0.0

out_dir = f"/kaggle/input/models/effdet-effdet_id-{effdet_id}-out_ids-{len(out_ids)}-channels-{channels}-max_size-{max_size}/{fold_idx}"
