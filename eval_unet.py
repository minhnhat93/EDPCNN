import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import torch
from train_unet import make_batch_input
from losses import dice_score, dice_score_inside_bboxes
import torch.nn.functional as F
import argparse
import numpy as np
from data_iterator import Dataset
from network import UNet
import _pickle


def do_eval(net, imgs, masks, bboxes, batch_sz):
    net.eval()
    with torch.no_grad():
        data_sz = len(imgs)
        n_batches = int(np.ceil(data_sz / batch_sz))
        dices_scores = []
        dices_scores_inside_bboxes = []
        for j in range(n_batches):
            start = j * batch_sz
            end = (j + 1) * batch_sz
            imgs_batch = imgs[start:end]
            masks_batch = masks[start:end]
            bboxes_batch = bboxes[start:end]
            imgs_batch = make_batch_input(imgs_batch)
            imgs_batch = torch.cuda.FloatTensor(imgs_batch)
            masks_batch = torch.cuda.FloatTensor(masks_batch)
            logits = net(imgs_batch)
            softmax = F.softmax(logits, dim=1)
            pred_masks = torch.argmax(softmax, dim=1)
            dices = dice_score(pred_masks, masks_batch, False)
            dices_bboxes = dice_score_inside_bboxes(pred_masks, masks_batch, bboxes_batch, False)
            dices_scores.append(dices)
            dices_scores_inside_bboxes.append(dices_bboxes)
        dices_scores = torch.cat(dices_scores, 0)
        dices_scores = torch.mean(dices_scores)
        dices_scores_inside_bboxes = torch.cat(dices_scores_inside_bboxes, 0)
        dices_scores_inside_bboxes = torch.mean(dices_scores_inside_bboxes)
    return dices_scores, dices_scores_inside_bboxes


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--batch_sz', type=int, default=10, metavar='N')
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

    print("eval...")
    dice, dice_inside_bboxes = do_eval(net, valid.images, valid.masks, valid.bboxes, args.batch_sz)
    print(f"dp valid dice: {dice}")
    print(f"dp valid dice inside bboxes: {dice_inside_bboxes}")

