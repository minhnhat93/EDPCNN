from snake.snake import SnakePytorch
import torch
from train_edpcnn import get_star_pattern_values, make_batch_input, smooth_ind, np, \
    dice_score, star_pattern_ind_to_mask, get_random_jitter
from losses import dice_score_inside_bboxes
import argparse
from data_iterator import Dataset
from network import UNet
import _pickle


def do_eval(net, snake: SnakePytorch, imgs, masks, centers, bboxes, batch_sz, num_lines, radius, smoothing_window=None):
    net.eval()
    H, W = imgs.shape[-2:]
    if smoothing_window is None:
        smoothing_window = num_lines // 4
    with torch.no_grad():
        data_sz = len(imgs)
        n_batches = int(np.ceil(data_sz / batch_sz))
        dices = []
        dices_inside_bboxes = []
        gs_fixed_shape = torch.zeros((snake.b_sz, num_lines, radius)).cuda()
        for j in range(n_batches):
            start = j * batch_sz
            end = (j + 1) * batch_sz
            imgs_batch = imgs[start:end]
            masks_batch = masks[start:end]
            centers_batch = centers[start:end]
            bboxes_batch = bboxes[start:end]
            batch_input = make_batch_input(imgs_batch)
            batch_input = torch.cuda.FloatTensor(batch_input)
            masks_batch = torch.cuda.FloatTensor(masks_batch)

            gs_logits = net(batch_input)
            gs_logits = gs_logits[:, 1, ...]

            gs_logits, _, _ = get_star_pattern_values(gs_logits, None, centers_batch, num_lines, radius + 1)

            gs = gs_logits[:, :, 1:] - gs_logits[:, :, :-1]

            # run DP algo
            # can only put batch with fixed shape into the snake algorithm
            gs_fixed_shape.fill_(0)
            gs_fixed_shape[:len(gs), ...] = gs
            ind_sets = snake(gs_fixed_shape).data.cpu().numpy()
            ind_sets = ind_sets[:len(gs), ...]
            ind_sets = np.expand_dims(smooth_ind(ind_sets.squeeze(-1), smoothing_window), -1)
            pred_masks = star_pattern_ind_to_mask(ind_sets, centers_batch, H, W, num_lines, radius)
            pred_masks = torch.cuda.FloatTensor(pred_masks)

            scores = dice_score(pred_masks, masks_batch, False)
            scores_inside_bboxes = dice_score_inside_bboxes(pred_masks, masks_batch, bboxes_batch, False)
            dices.append(scores)
            dices_inside_bboxes.append(scores_inside_bboxes)
        dices = torch.cat(dices, 0)
        dices = torch.mean(dices)
        dices_inside_bboxes = torch.cat(dices_inside_bboxes, 0)
        dices_inside_bboxes = torch.mean(dices_inside_bboxes)
    return dices, dices_inside_bboxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--delta', type=int, default=1, metavar='N',
                        help='smoothness constraint for snake')
    parser.add_argument('--num_lines', type=int, default=25, metavar='N',
                        help='number of radial lines')
    parser.add_argument('--radius', type=int, default=65, metavar='N',
                        help='radius of the star pattern, should be big'
                             ' enough to cover the segmentation patches')
    parser.add_argument('--smoothing_window', default=7, type=int)
    parser.add_argument('--batch_sz', type=int, default=20, metavar='N')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    d = Dataset(train_set_size=100)
    train = d.train_set
    valid = d.test_set

    H, W = train.images.shape[-2:]
    num_classes = 1
    net = UNet(in_dim=1, out_dim=num_classes + 1).cuda()

    net.load_state_dict(_pickle.load(open(args.ckpt_path, 'rb')))

    snake = SnakePytorch(args.delta, args.batch_sz, args.num_lines, args.radius)

    print("eval...")
    dice, dice_inside_bboxes = \
        do_eval(net, snake, valid.images, valid.masks, valid.centers, valid.bboxes, args.batch_sz, args.num_lines,
                args.radius, smoothing_window=args.smoothing_window)
    print(f"dp valid dice: {dice}")
    print(f"dp valid dice inside bboxes: {dice_inside_bboxes}")