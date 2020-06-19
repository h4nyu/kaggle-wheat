label_path = "/store/train.csv"
plot_dir = "/store/plot"
image_dir = "/store/images"
submition_csv = "/store/sample_submissions.csv"
root_dir = "/store"

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
num_workers: int = 8
scale_factor = 2
