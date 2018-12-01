import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


# =============================================
# U-Net
# modified from: https://raw.githubusercontent.com/zijundeng/pytorch-semantic-segmentation/master/models/u_net.py

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, n_bc=64, in_dim=3, out_dim=1):
        super(UNet, self).__init__()
        self.pad = nn.ConstantPad2d(92, 0.0)
        self.enc1 = _EncoderBlock(in_dim, n_bc)
        self.enc2 = _EncoderBlock(n_bc, n_bc * 2)
        self.enc3 = _EncoderBlock(n_bc * 2, n_bc * 4)
        self.enc4 = _EncoderBlock(n_bc * 4, n_bc * 8, dropout=True)
        self.center = _DecoderBlock(n_bc * 8, n_bc * 16, n_bc * 8)
        self.dec4 = _DecoderBlock(n_bc * 16, n_bc * 8, n_bc * 4)
        self.dec3 = _DecoderBlock(n_bc * 8, n_bc * 4, n_bc * 2)
        self.dec2 = _DecoderBlock(n_bc * 4, n_bc * 2, n_bc)
        self.dec1 = nn.Sequential(
            nn.Conv2d(n_bc * 2, n_bc, kernel_size=3),
            nn.BatchNorm2d(n_bc),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bc, n_bc, kernel_size=3),
            nn.BatchNorm2d(n_bc),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(n_bc, out_dim, kernel_size=1)  # 2 out channel = mean + variance
        initialize_weights(self)

    def forward(self, x):
        pad = self.pad(x)
        enc1 = self.enc1(pad)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return final


# ============================================
# Approximate network
class _EncoderBlockWithPadding(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlockWithPadding, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlockWithPadding(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlockWithPadding, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class SnakeApproxNet(nn.Module):
    def __init__(self, in_dim=1, n_bc=64):
        super(SnakeApproxNet, self).__init__()
        out_dim = 1
        self.enc1 = _EncoderBlockWithPadding(in_dim, n_bc)
        self.enc2 = _EncoderBlockWithPadding(n_bc, n_bc * 2)
        self.enc3 = _EncoderBlockWithPadding(n_bc * 2, n_bc * 4, dropout=True)
        self.center = _DecoderBlockWithPadding(n_bc * 4, n_bc * 8, n_bc * 4)
        self.dec3 = _DecoderBlockWithPadding(n_bc * 8, n_bc * 4, n_bc * 2)
        self.dec2 = _DecoderBlockWithPadding(n_bc * 4, n_bc * 2, n_bc)
        self.dec1 = nn.Sequential(
            nn.Conv2d(n_bc * 2, n_bc, kernel_size=3),
            nn.BatchNorm2d(n_bc),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bc, n_bc, kernel_size=3),
            nn.BatchNorm2d(n_bc),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(n_bc, out_dim, kernel_size=1)  # 2 out channel = mean + variance
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.interpolate(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        up_sample = F.interpolate(final, x.size()[-2:], mode='bilinear')
        return up_sample
