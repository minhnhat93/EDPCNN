from train_edpcnn import parse_args, main as train
import os
import numpy as np

train_set_sz_list = reversed([10, 20, 50, 100, 200, 500, 1000, 1436])
n_iters_list = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000]

for n_iters, train_set_sz in zip(n_iters_list, train_set_sz_list):
    args = parse_args()
    args.train_set_sz = train_set_sz
    args.lr = 1e-4
    args.delta = 2
    args.batch_sz = 10
    args.log_freq = 1
    args.num_lines = 50
    args.radius = 65
    args.snake_batch_sz = 100
    eval_freq = 50
    args.train_eval_freq = eval_freq
    args.val_eval_freq = eval_freq
    args.sigma_scaling = 1
    args.gs_decay = 0.00
    args.n_epochs = int(np.ceil(n_iters / (train_set_sz / args.batch_sz)))
    args.smoothing_window = 5
    args.dice_approx_train_steps = 10
    args.num_samples = 10
    args.use_center_jitter = True
    args.theta_jitter = np.pi
    args.log_dir = os.path.join(f"log/edpcnn/"
                                f"train_set_sz={train_set_sz},delta=2,smooth=5,nl=50,theta_jitter=pi")
    print(args)
    val_dice = train(args)
