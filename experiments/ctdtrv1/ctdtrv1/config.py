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
batch_size = 11
num_workers = 8

# model
effdet_id:Phi = 4
depth = 1
out_idx: PyramidIdx = 5
channels = 128
pretrained = True

sigma = 5.0
heatmap_weight = 1.0
box_weight = 10.0
iou_threshold = 0.50
use_peak = True

confidence_threshold = 0.32
final_threshold = 0.0

fpn_depth = 1
hm_depth = 1
box_depth = 1

out_dir = f"/kaggle/input/models/ctdtv1-effdet_id-{effdet_id}-fpn_depth-{fpn_depth}-hm_depth-{hm_depth}-box_depth-{box_depth}-channels-{channels}-out_idx-{out_idx}-max_size-{max_size}/{fold_idx}"
