from network import UNet, SnakeApproxNet
import torch
import argparse
from data_iterator import Dataset
from utils import seed_all, normalize
import os
import tensorboardX
from losses import dice_score, dice_loss
import torch.nn.functional as F
import numpy as np
import _pickle
from scipy.signal import convolve2d
from snake.snake import SnakePytorch
from snake.star_pattern_utils import star_pattern_ind_to_image_ind
from snake.utils import bilinear_inter
import cv2
from scipy.stats.distributions import truncnorm
import timeit


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--delta', type=int, default=5, metavar='N',
                        help='hard smoothness constraint for snake')
    parser.add_argument('--num_lines', type=int, default=25, metavar='N',
                        help='number of radial lines')
    parser.add_argument('--use_center_jitter', dest='use_center_jitter', action='store_true')
    parser.add_argument('--no_use_center_jitter', dest='use_center_jitter', action='store_false')
    parser.set_defaults(use_center_jitter=True)
    parser.add_argument('--theta_jitter', type=float, default=0.5)
    parser.add_argument('--radius', type=float, default=65, metavar='N',
                        help='radius of the star pattern, should be big'
                             ' enough to cover the segmentation patches')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_epochs', type=int, default=1000, metavar='N')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--smoothing_window', type=int, default=7)
    parser.add_argument('--batch_sz', type=int, default=10, metavar='N')
    parser.add_argument('--log_freq', type=int, default=1, metavar='N')
    parser.add_argument('--train_eval_freq', default=50, type=int)
    parser.add_argument('--val_eval_freq', default=100, type=int)
    parser.add_argument('--train_set_sz', default=100000000, type=int)
    parser.add_argument('--log_dir', default='log/sg', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--dice_approx_train_steps', type=int, default=10, metavar='N',
                        help='number of time to train dice approx network each iteration')
    parser.add_argument('--num_samples', type=int, default=10, metavar='N',
                        help='number of random sample use to train dice approx.'
                             'note: randomly sample and retrain so that the dice loss approx network doesnt'
                             'get overfitted')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for the random process')
    args = parser.parse_args()
    return args


def make_batch_input(imgs):
    xs = []
    for j, img in enumerate(imgs):
        xs.append(img)
    xs = np.asarray(xs)
    return xs


def star_pattern_ind_to_mask(ind_sets, centers, H, W, num_lines, radius, center_jitters=None, angle_jitters=None,
                             num_samples=1):
    if center_jitters is None:
        center_jitters = [None] * len(ind_sets)
    if angle_jitters is None:
        angle_jitters = [None] * len(ind_sets)
    masks = []
    for j in range(len(ind_sets)):
        inds = ind_sets[j]
        center = centers[j // num_samples]  # repeated center
        c_o = center_jitters[j // num_samples]
        a_o = angle_jitters[j // num_samples]
        contour = star_pattern_ind_to_image_ind(
            inds, center, 1, radius, num_lines, radius, round_to_int=True,
            max_dim=[H - 1, W - 1],
            center_jitter=c_o, angle_jitter=a_o
        )
        contour = contour[:, ::-1]
        mask = np.zeros((H, W))
        cv2.fillPoly(mask, pts=[contour.astype(int)], color=(1,))
        masks.append(mask)
    masks = np.asarray(masks)
    return masks


def get_star_pattern_values(vs_whole_img, selected_ind_sets, centers, num_lines, radius, center_jitters=None,
                            angle_jitters=None, use_bilinear_inter=False, num_samples=1):
    vs = []
    coords_r, coords_c = [], []
    if selected_ind_sets is None:
        selected_ind_sets = [None] * len(vs_whole_img)
        v_shape = (1, num_lines, radius)
    else:
        v_shape = (1, num_lines, 1)

    if center_jitters is None:
        center_jitters = [None] * len(vs_whole_img)
    if angle_jitters is None:
        angle_jitters = [None] * len(vs_whole_img)

    H, W = list(vs_whole_img.shape[-2:])
    for j in range(len(vs_whole_img)):
        v_whole_img = vs_whole_img[j]
        selected_ind_set = selected_ind_sets[j]
        center = centers[j // num_samples]  # repeated by num_samples
        c_j = center_jitters[j // num_samples]
        a_j = angle_jitters[j // num_samples]
        coord = star_pattern_ind_to_image_ind(
            selected_ind_set, center, 1, radius, num_lines, radius, round_to_int=False,
            max_dim=[H - 1, W - 1],
            center_jitter=c_j, angle_jitter=a_j,
        )
        coord_r, coord_c = coord[:, 0].flatten(), coord[:, 1].flatten()
        if use_bilinear_inter:
            v = bilinear_inter(coord_r, coord_c, v_whole_img)
        else:
            v = v_whole_img[..., coord_r, coord_c]
        v = v.reshape(v_shape)
        vs.append(v)
        coords_r.append(coord_r)
        coords_c.append(coord_c)
    vs = torch.cat(vs, dim=0)
    coords_r = np.asarray(coords_r)
    coords_c = np.asarray(coords_c)
    return vs, coords_r, coords_c


def smooth_ind(ind_sets, window_sz=11):
    # from running snake on true distance transform: best window_sz = 5
    mask = np.asarray([[1] * window_sz]) / window_sz
    return convolve2d(ind_sets, mask, "same", "wrap")


def do_eval(net, snake: SnakePytorch, imgs, masks, centers, batch_sz, num_lines, radius, smoothing_window=None):
    net.eval()
    H, W = imgs.shape[-2:]
    if smoothing_window is None:
        smoothing_window = num_lines // 4
    with torch.no_grad():
        data_sz = len(imgs)
        n_batches = int(np.ceil(data_sz / batch_sz))
        total_loss = []
        gs_fixed_shape = torch.zeros((snake.b_sz, num_lines, radius)).cuda()
        for j in range(n_batches):
            start = j * batch_sz
            end = (j + 1) * batch_sz
            imgs_batch = imgs[start:end]
            masks_batch = masks[start:end]
            centers_batch = centers[start:end]
            batch_input = make_batch_input(imgs_batch)
            batch_input = torch.cuda.FloatTensor(batch_input)
            masks_batch = torch.cuda.FloatTensor(masks_batch)

            gs_logits = net(batch_input)

            # get pixel values on the star pattern
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

            loss = dice_score(pred_masks, masks_batch, False)
            total_loss.append(loss)
        total_loss = torch.cat(total_loss, 0)
        total_loss = torch.mean(total_loss)
    return total_loss


def get_random_jitter(radius, theta, num=1):
    c_js, a_js = [], []
    for _ in range(num):
        c_j = sample_2d_truncnorm(radius, 1).flatten()
        a_j = np.random.uniform(-theta, theta)
        c_js.append(c_j)
        a_js.append(a_j)
    return c_js, a_js


def sample_2d_truncnorm(radius, size):
    radius = truncnorm.rvs(-1, 1, loc=0, scale=radius, size=size)
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    out = np.stack((x, y), -1)
    return out


def mask_to_indices(mask, center, radius, num_lines, center_jitter, angle_jitter):
    from snake.star_pattern_utils import star_pattern_to_segments, find_intersection
    from skimage.measure import find_contours
    segments = star_pattern_to_segments(center, radius, num_lines, center_jitter, angle_jitter)
    contour = find_contours(mask, 0.8)[0]  # output of this function is (row, col)
    indices = find_intersection(segments, contour, 1, radius)
    indices = indices * radius
    return indices


def main(args):
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)

    num_classes = 1

    d = Dataset(train_set_size=args.train_set_sz, num_cls=num_classes)
    train = d.train_set
    valid = d.test_set

    net = UNet(in_dim=1, out_dim=num_classes).cuda()
    snake_approx_net = SnakeApproxNet(n_bc=8).cuda()
    best_val_dice = -np.inf

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    snake_approx_optimizer = torch.optim.Adam(params=snake_approx_net.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay)

    # load model
    if args.ckpt:
        loaded = _pickle.load(open(args.ckpt, 'rb'))
        net.load_state_dict(loaded[0])
        optimizer.load_state_dict(loaded[1])
        snake_approx_net.load_state_dict(loaded[2])
        snake_approx_optimizer.load_state_dict(loaded[3])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)
    snake = SnakePytorch(args.delta, args.batch_sz * args.num_samples, args.num_lines, args.radius)
    snake_eval = SnakePytorch(args.delta, args.batch_sz, args.num_lines, args.radius)
    noises = torch.zeros((args.batch_sz, args.num_samples, args.num_lines, args.radius)).cuda()

    step = 1
    start = timeit.default_timer()
    for epoch in range(1, args.n_epochs + 1):
        for iteration in range(1, int(np.ceil(train.dataset_sz() / args.batch_sz)) + 1):

            imgs, masks, onehot_masks, centers, dts_modified, dts_original, jitter_radius, bboxes = \
                train.next_batch(args.batch_sz)

            xs = make_batch_input(imgs)
            xs = torch.cuda.FloatTensor(xs)

            net.train()
            unet_logits = net(xs)

            center_jitters, angle_jitters = [], []
            for img, center, j_r in zip(imgs, centers, jitter_radius):
                if not args.use_center_jitter:
                    j_r = 0
                c_j, a_j = get_random_jitter(j_r, args.theta_jitter)
                center_jitters.append(c_j)
                angle_jitters.append(a_j)

            center_jitters = np.asarray(center_jitters)
            angle_jitters = np.asarray(angle_jitters)

            # args.radius + 1 because we need additional outermost points for the gradient
            gs_logits_whole_img = unet_logits
            gs_logits, coords_r, coords_c = get_star_pattern_values(gs_logits_whole_img, None, centers, args.num_lines,
                                                                    args.radius + 1, center_jitters=center_jitters,
                                                                    angle_jitters=angle_jitters)

            # currently only class 1 is foreground
            # if there's multiple foreground classes use a for loop
            gs = gs_logits[:, :, 1:] - gs_logits[:, :, :-1]  # compute the gradient

            noises.normal_(0, 1)  # noises here is only used for random exploration so no need mirrored sampling
            gs_noisy = torch.unsqueeze(gs, 1) + noises

            def batch_eval_snake(snake, inputs, batch_sz):
                n_inputs = len(inputs)
                assert n_inputs % batch_sz == 0
                n_batches = int(np.ceil(n_inputs / batch_sz))
                ind_sets = []
                for j in range(n_batches):
                    inps = inputs[j * batch_sz: (j + 1) * batch_sz]
                    batch_ind_sets = snake(inps).data.cpu().numpy()
                    ind_sets.append(batch_ind_sets)
                ind_sets = np.concatenate(ind_sets, 0)
                return ind_sets

            gs_noisy = gs_noisy.reshape((args.batch_sz * args.num_samples, args.num_lines, args.radius))
            ind_sets = batch_eval_snake(snake, gs_noisy, args.batch_sz * args.num_samples)
            ind_sets = ind_sets.reshape((args.batch_sz * args.num_samples, args.num_lines))
            ind_sets = np.expand_dims(smooth_ind(ind_sets, args.smoothing_window), -1)

            # loss layers
            m = torch.nn.LogSoftmax(dim=1)
            loss = torch.nn.NLLLoss()

            # ===========================================================================
            # Inner loop: Train dice loss prediction network
            snake_approx_net.train()
            for _ in range(args.dice_approx_train_steps):

                snake_approx_logits = snake_approx_net(
                    gs_noisy.reshape(args.batch_sz * args.num_samples, 1, args.num_lines, args.radius).detach())
                snake_approx_train_loss = loss(m(snake_approx_logits.squeeze().transpose(2, 1)),
                                               torch.cuda.LongTensor(ind_sets.squeeze()))
                snake_approx_optimizer.zero_grad()
                snake_approx_train_loss.backward()
                snake_approx_optimizer.step()
            # ===========================================================================

            # ===========================================================================
            # Now, minimize the approximate dice loss
            snake_approx_net.eval()

            gt_indices = []
            for mask, center, cj, aj in zip(masks, centers, center_jitters, angle_jitters):
                gt_ind = mask_to_indices(mask, center, args.radius, args.num_lines, cj, aj)
                gt_indices.append(gt_ind)
            gt_indices = np.asarray(gt_indices).astype(int)

            gt_indices = gt_indices.reshape((args.batch_sz, args.num_lines))
            gt_indices = torch.cuda.LongTensor(gt_indices)

            snake_approx_logits = snake_approx_net(gs.reshape((args.batch_sz, 1, args.num_lines, args.radius)))
            nll_approx_loss = loss(m(snake_approx_logits.squeeze().transpose(2, 1)), gt_indices)

            total_loss = nll_approx_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # ===========================================================================

            snake_approx_train_loss = snake_approx_train_loss.data.cpu().numpy()
            nll_approx_loss = nll_approx_loss.data.cpu().numpy()
            total_loss = snake_approx_train_loss + nll_approx_loss

            if step % args.log_freq == 0:
                stop = timeit.default_timer()
                print(f"step={step}\tepoch={epoch}\titer={iteration}"
                      f"\tloss={total_loss}"
                      f"\tsnake_approx_train_loss={snake_approx_train_loss}"
                      f"\tnll_approx_loss={nll_approx_loss}"
                      f"\tlr={optimizer.param_groups[0]['lr']}"
                      f"\ttime={stop-start}")
                start = stop
                writer.add_scalar("total_loss", total_loss, step)
                writer.add_scalar("nll_approx_loss", nll_approx_loss, step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

            if step % args.train_eval_freq == 0:
                train_dice = do_eval(net, snake, train.images, train.masks, train.centers, args.batch_sz,
                                     args.num_lines, args.radius,
                                     smoothing_window=args.smoothing_window).data.cpu().numpy()
                writer.add_scalar("train_dice", train_dice, step)
                print(f"step={step}\tepoch={epoch}\titer={iteration}\ttrain_eval: train_dice={train_dice}")

            if step % args.val_eval_freq == 0:
                val_dice = do_eval(net, snake_eval, valid.images, valid.masks, valid.centers, args.batch_sz,
                                   args.num_lines, args.radius,
                                   smoothing_window=args.smoothing_window).data.cpu().numpy()
                writer.add_scalar("val_dice", val_dice, step)
                print(f"step={step}\tepoch={epoch}\titer={iteration}\tvalid_dice={val_dice}")
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    _pickle.dump([net.state_dict(), optimizer.state_dict(),
                                  snake_approx_net.state_dict(), snake_approx_optimizer.state_dict()],
                                 open(os.path.join(args.log_dir, 'best_model.pth.tar'), 'wb'))
                    f = open(os.path.join(args.log_dir, f"best_val_dice{step}.txt"), 'w')
                    f.write(str(best_val_dice))
                    f.close()
                    print(f"better val dice detected.")

            step += 1

    return best_val_dice


if __name__ == "__main__":
    args = parse_args()
    main(args)
