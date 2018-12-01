from train_sg import parse_args, main as train
import os

n_lines_list = [12, 25, 50, 100]
delta_list = [1, 2, 5, 7, 10]

for num_lines in n_lines_list:
    for delta in delta_list:
        args = parse_args()
        args.train_set_sz = 1436
        args.lr = 1e-4
        args.delta = delta
        args.batch_sz = 10
        args.log_freq = 1
        args.num_lines = num_lines
        args.radius = 65
        args.snake_batch_sz = 100
        eval_freq = 50
        args.train_eval_freq = eval_freq
        args.val_eval_freq = eval_freq
        args.n_epochs = 200
        args.smoothing_window = num_lines // 4 + 1
        args.dice_approx_train_steps = 10
        args.num_samples = 10
        args.use_center_jitter = True
        args.log_dir = os.path.join(f"log/sg_param_test/"
                                    f"num_lines={num_lines},delta={delta},full_dataset")
        print(args)
        val_dice = train(args)
