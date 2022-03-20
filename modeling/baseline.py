import torch
from torch import nn
import torch.nn.functional as F
import sys
import pdb

from .backbones.se_module import SELayer

sys.path.append('.')

EPSILON = 1e-12


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class SAMS(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """

    def __init__(self, in_channels, channels, radix=4, reduction_factor=4, norm_layer=nn.BatchNorm2d):
        super(SAMS, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=1)

    def forward(self, x):

        return out.contiguous()


class SELayer(nn.Module):  # channel-wise attention
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, level):
        super(Baseline, self).__init__()
        print(f"Training with pyramid level {level}")
        self.level = level

        self.base_1 = nn.Conv2d(256, 64, kernel_size=3, padding=0, bias=False)
        self.base_2 = nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False)
        self.base_3 = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False)
        self.base_4 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, bias=False)
        self.base_5 = nn.Conv2d(1024, 2048, kernel_size=1, padding=0, bias=False)

        if self.level > 0:
            self.att1 = SELayer(64, 8)
            self.att2 = SELayer(256, 32)
            self.att3 = SELayer(512, 64)
            self.att4 = SELayer(1024, 128)
            self.att5 = SELayer(2048, 256)
            if self.level > 1:  # second pyramid level
                self.att_s1 = SAMS(64, int(64 / self.level), radix=self.level)
                self.att_s2 = SAMS(256, int(256 / self.level), radix=self.level)
                self.att_s3 = SAMS(512, int(512 / self.level), radix=self.level)
                self.att_s4 = SAMS(1024, int(1024 / self.level), radix=self.level)
                self.att_s5 = SAMS(2048, int(2048 / self.level), radix=self.level)
                if self.level > 2:
                    raise RuntimeError("We do not support pyramid level greater than TWO.")

        self.output = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, padding=0, bias=False),
                                    nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
                                    nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False))

    def forward(self, x):

        x = self.base_1(x)
        if self.level > 1:
            x = self.att_s1(x)
        if self.level > 0:
            y = self.att1(x)
            x = x * y.expand_as(x)
            # level_0_weight_v = self.weight_level_0(x)

        x = self.base_2(x)
        if self.level > 1:
            x = self.att_s2(x)
        if self.level > 0:
            y = self.att2(x)
            x = x * y.expand_as(x)
            # level_0_weight_v = self.weight_level_0(x)

        x = self.base_3(x)
        if self.level > 1:
            x = self.att_s3(x)
        if self.level > 0:
            y = self.att3(x)
            x = x * y.expand_as(x)
            # level_0_weight_v = self.weight_level_0(x)

        x = self.base_4(x)
        if self.level > 1:
            x = self.att_s4(x)
        if self.level > 0:
            y = self.att4(x)
            x = x * y.expand_as(x)
            # level_0_weight_v = self.weight_level_0(x)

        x = self.base_5(x)
        if self.level > 1:
            x = self.att_s5(x)
        if self.level > 0:
            y = self.att5(x)
            x = x * y.expand_as(x)
            # level_0_weight_v = self.weight_level_0(x)

        return self.output(x)
