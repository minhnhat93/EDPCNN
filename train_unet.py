from network import UNet
import torch
import argparse
from data_iterator import Dataset
from utils import seed_all, one_hot_embedding_pytorch, normalize
import os
import tensorboardX
from losses import dice_loss, dice_score
import torch.nn.functional as F
import numpy as np
import _pickle
import timeit


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_epochs', type=int, default=1000, metavar='N')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_sz', type=int, default=10, metavar='N')
    parser.add_argument('--log_freq', type=int, default=50, metavar='N')
    parser.add_argument('--train_eval_freq', default=200, type=int)
    parser.add_argument('--val_eval_freq', default=100, type=int)
    parser.add_argument('--train_set_sz', default=100000000, type=int)
    parser.add_argument('--num_cls', default=1, type=int)
    parser.add_argument('--log_dir', default='log/unet', type=str)
    args = parser.parse_args()
    return args


def make_batch_input(imgs):
    xs = []
    for j, img in enumerate(imgs):
        x = img
        xs.append(x)
    xs = np.asarray(xs)
    return xs


def do_eval(net, imgs, one_hot_masks, batch_sz, num_classes):
    net.eval()
    with torch.no_grad():
        data_sz = len(imgs)
        n_batches = int(np.ceil(data_sz / batch_sz))
        dice_scores = []
        for j in range(n_batches):
            start = j * batch_sz
            end = (j + 1) * batch_sz
            imgs_batch = imgs[start:end]
            imgs_batch = make_batch_input(imgs_batch)
            imgs_batch = torch.cuda.FloatTensor(imgs_batch)
            one_hot_batch = one_hot_masks[start:end]
            one_hot_batch = torch.cuda.FloatTensor(one_hot_batch)
            logits = net(imgs_batch)
            softmax = F.softmax(logits, dim=1)
            pred_class = torch.argmax(softmax, dim=1)
            pred_one_hot = one_hot_embedding_pytorch(pred_class, num_classes).permute([0, 3, 1, 2])
            loss = dice_score(pred_one_hot, one_hot_batch)
            dice_scores.append(loss)
        dice_scores = torch.cat(dice_scores, 0)
        cls_dice_scores = torch.mean(dice_scores, 0)
        mean_dice_score = torch.mean(cls_dice_scores)
    return mean_dice_score, cls_dice_scores


def main(args):
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)

    d = Dataset(train_set_size=args.train_set_sz, num_cls=args.num_cls)
    train = d.train_set
    valid = d.test_set

    num_cls = args.num_cls + 1  # +1 for background
    net = UNet(in_dim=1, out_dim=num_cls).cuda()
    best_net = UNet(in_dim=1, out_dim=num_cls)
    best_val_dice = -np.inf
    best_cls_val_dices = None

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    step = 1
    for epoch in range(1, args.n_epochs + 1):
        for iteration in range(1, int(np.ceil(train.dataset_sz() / args.batch_sz)) + 1):

            net.train()

            imgs, masks, one_hot_masks, centers, _, _, _, _ = train.next_batch(args.batch_sz)
            imgs = make_batch_input(imgs)
            imgs = torch.cuda.FloatTensor(imgs)
            one_hot_masks = torch.cuda.FloatTensor(one_hot_masks)

            pred_logit = net(imgs)
            pred_softmax = F.softmax(pred_logit, dim=1)

            loss = dice_loss(pred_softmax, one_hot_masks, keep_background=False).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_freq == 0:
                print(f"step={step}\tepoch={epoch}\titer={iteration}\tdice_loss={loss.data.cpu().numpy()}")
                writer.add_scalar("cnn_dice_loss", loss.data.cpu().numpy(), step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

            if step % args.train_eval_freq == 0:
                train_dice, cls_train_dices = do_eval(net, train.images, train.onehot_masks, args.batch_sz, num_cls)
                train_dice = train_dice.cpu().numpy()
                cls_train_dices = cls_train_dices.cpu().numpy()
                writer.add_scalar("train_dice", train_dice, step)
                for j, cls_train_dice in enumerate(cls_train_dices):
                    writer.add_scalar(f"train_dice/{j}", cls_train_dice, step)
                print(f"step={step}\tepoch={epoch}\titer={iteration}\ttrain_eval: train_dice={train_dice}")

            if step % args.val_eval_freq == 0:
                _pickle.dump(net.state_dict(), open(os.path.join(args.log_dir, 'model.pth.tar'), 'wb'))
                val_dice, cls_val_dices = do_eval(net, valid.images, valid.onehot_masks, args.batch_sz, num_cls)
                val_dice = val_dice.cpu().numpy()
                cls_val_dices = cls_val_dices.cpu().numpy()
                writer.add_scalar("val_dice", val_dice, step)
                for j, cls_val_dice in enumerate(cls_val_dices):
                    writer.add_scalar(f"val_dice/{j}", cls_val_dice, step)
                print(f"step={step}\tepoch={epoch}\titer={iteration}\tvalid_dice={val_dice}")
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    best_cls_val_dices = cls_val_dices
                    best_net.load_state_dict(net.state_dict().copy())
                    _pickle.dump(best_net.state_dict(), open(os.path.join(args.log_dir, 'best_model.pth.tar'), 'wb'))
                    f = open(os.path.join(args.log_dir, f"best_val_dice{step}.txt"), 'w')
                    f.write(str(best_val_dice) + "\n")
                    f.write(" ".join([str(dice_score) for dice_score in best_cls_val_dices]))
                    f.close()
                    print(f"better val dice detected.")

            step += 1

    return best_val_dice, best_cls_val_dices


if __name__ == "__main__":
    args = parse_args()
    main(args)
