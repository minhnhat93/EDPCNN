from train_unet import parse_args as parse_unet_args, main as train_unet
import os
import numpy as np


train_set_sz_list = reversed([10, 20, 50, 100, 200, 500, 1000, 1516])
n_iters_list = [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000]


val_dice_curve = []
for n_iters, train_set_sz in zip(n_iters_list, train_set_sz_list):
    for weight_decay in [0.0]:
        args = parse_unet_args()
        args.train_set_sz = train_set_sz
        args.batch_sz = 10
        args.n_epochs = int(np.ceil(n_iters / (train_set_sz / args.batch_sz)))
        args.num_cls = 1
        args.lr = 1e-4
        eval_freq = 50
        args.train_eval_freq = eval_freq
        args.val_eval_freq = eval_freq
        args.use_ce = True
        args.log_dir = "log/unet-1_cls/sz={}".format(train_set_sz)
        val_dice, _ = train_unet(args)
