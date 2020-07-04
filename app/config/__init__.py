import os

label_path = "/kaggle/train.csv"
plot_dir = "/kaggle/plot"
image_dir = "/kaggle/images"
submition_csv = "/kaggle/sample_submission.csv"
root_dir = "/kaggle"
output_dir = "/kaggle/working"
model_dir = "/kaggle/input/models"
annot_file = "/kaggle/input/global-wheat-detection/train.csv"
working_dir = "/kaggle/working"
test_image_dir = "/kaggle/input/global-wheat-detection/test"
train_image_dir = "/kaggle/input/global-wheat-detection/train"

max_size = 512

device = "cuda"
random_state = 777
lr = 1e-3
n_splits = 5

num_classes = 1
num_queries = 50
hidden_dim = 128

eos_coef = 0.01
loss_label = 1
loss_box = 5
loss_giou = 0

cost_class = 1
cost_box = 1
cost_giou = 1

batch_size: int = 8
no_grad_batch_size: int = 16
num_workers: int = min(batch_size, os.cpu_count())  # type:ignore
scale_factor = 2
