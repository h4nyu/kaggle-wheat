from object_detection.entities import PyramidIdx
import torch

test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"

n_splits = 4
fold_idx = 0
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

max_size = 512
batch_size = 12
num_workers = 8

depth = 1
out_idx: PyramidIdx = 4
channels = 64

sigma = 6.0
heatmap_weight = 1.0
box_weight = 50.0
iou_threshold = 0.5
use_peak = True

confidence_threshold = 0.3

fpn_depth = 1
hm_depth = 1
box_depth = 1

out_dir = f"/kaggle/input/models/ctdtv1-fpn_depth-{fpn_depth}-hm_depth-{hm_depth}-box_depth-{box_depth}-channels-{channels}-out_idx-{out_idx}-max_size-{max_size}/{fold_idx}"
