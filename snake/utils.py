import torch
import numpy as np


def bilinear_inter(r, c, im):
    H, W = im.shape[-2:]
    r0 = r.astype(np.int)
    r1 = r0 + 1
    c0 = c.astype(np.int)
    c1 = c0 + 1
    r = torch.cuda.FloatTensor(np.clip(r, 0, H - 1))
    r0 = torch.cuda.LongTensor(np.clip(r0, 0, H - 1))
    r1 = torch.cuda.LongTensor(np.clip(r1, 0, H - 1))
    c = torch.cuda.FloatTensor(np.clip(c, 0, W - 1))
    c0 = torch.cuda.LongTensor(np.clip(c0, 0, W - 1))
    c1 = torch.cuda.LongTensor(np.clip(c1, 0, W - 1))
    interpolated_im = ((r - r0.type(torch.cuda.FloatTensor)) * (c - c0.type(torch.cuda.FloatTensor)) * im[..., r0, c0] +
                       (r1.type(torch.cuda.FloatTensor) - r) * (c - c0.type(torch.cuda.FloatTensor)) * im[..., r1, c0] +
                       (r - r0.type(torch.cuda.FloatTensor)) * (c1.type(torch.cuda.FloatTensor) - c) * im[..., r0, c1] +
                       (r1.type(torch.cuda.FloatTensor) - r) * (c1.type(torch.cuda.FloatTensor) - c) * im[..., r1, c1])
    return interpolated_im


def one_hot_encoding(x: torch.Tensor, depth=None, axis=-1):
    assert axis == -1, "other axis not implemented"
    x = x.type(torch.cuda.LongTensor)
    if depth is None:
        depth = x.max() + 1
    one_hot = torch.eye(depth).cuda()[x.flatten()]
    one_hot = one_hot.reshape(list(x.shape) + [-1])
    return one_hot
