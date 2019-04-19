import torch
import numpy as np
import random
from network import UNet
import tensorboardX


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot_embedding_pytorch(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    labels_shape = list(labels.shape)
    labels = labels.view(-1)
    y = torch.eye(num_classes).cuda()
    one_hot = y[labels].reshape(labels_shape + [num_classes])
    return one_hot


def normalize(x):
    return (x - x.mean()) / x.std()


def log_param_and_grad(net: UNet, writer: tensorboardX.SummaryWriter, step):
    for name, param in net.named_parameters():
        writer.add_histogram(f"grad/{name}", param.grad.detach().cpu().numpy(), step)
        writer.add_histogram(f"grad_norm/{name}", np.sqrt((param**2).sum().detach().cpu().numpy()), step)
        writer.add_histogram(f"param/{name}", param.detach().cpu().numpy(), step)

