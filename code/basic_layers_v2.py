from multiprocessing import pool
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import pdb

def make_convolution_layer(
    channel_input,
    channel_output,
    kernel=3,
    pool_type=None,
    dropout_rate=0.4):
    core_layer = [
        nn.Conv2d(channel_input, channel_output, kernel),
        nn.BatchNorm2d(channel_output),
        nn.ReLU()
    ]

    if not pool_type:
       core_layer += [nn.Dropout(p=dropout_rate)]

    if pool_type == 'max':
        pool_layer  = nn.MaxPool2d(kernel_size=3, stride=2)
        core_layer += [pool_layer, nn.Dropout(p=dropout_rate)]

    return nn.Sequential(*core_layer)


class ResidualBlock(nn.Module):

    def __init__(self, channel_input, upsample_size=None, channel_output=None, dropout_rate=0.4):

        super(ResidualBlock, self).__init__()
        if not channel_output:
            channel_output = channel_input
        self.conv_block1 = make_convolution_layer(channel_input, channel_output)
        self.conv_block2 = make_convolution_layer(channel_output, channel_output)
        self.conv_block3 = make_convolution_layer(channel_output, channel_output)
        if upsample_size:
            self.conv_upsample = nn.Sequential(
                nn.Conv2d(
                    channel_input,
                    channel_output,
                    kernel_size=3,
                    bias=False),
                nn.UpsamplingBilinear2d(
                    size=upsample_size
                )
            )

    def forward(self, x):
        residual = x
        out_c1 = self.conv_block1(x)
        out_c2 = self.conv_block2(out_c1)
        out_c3 = self.conv_block3(out_c2)
        if residual.shape != out_c3.shape:
            residual = self.conv_upsample(residual)
        return out_c3 + residual


class AttentionBlock1(nn.Module):
    def __init__(self, channel_input=64, channel_output=64, dropout_rate=0.4) -> None:
        super(AttentionBlock1, self).__init__()
        self.residual_one = ResidualBlock(64, (76, 76))
        self.trunk = nn.Sequential(
            ResidualBlock(64, (70, 70)),
            ResidualBlock(64, (64, 64)),
            nn.UpsamplingBilinear2d(
                size=(60, 60)
            )
        )

        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual_two = ResidualBlock(64, (31, 31))
        self.skip_one = nn.Sequential(
            ResidualBlock(64, (25, 25)),
            nn.UpsamplingBilinear2d(size=(60, 60))
        )
        self.upsample = nn.UpsamplingBilinear2d(size=(90, 90))
        self.activation_block = nn.Sequential(
            make_convolution_layer(64, 64),
            make_convolution_layer(64, 64),
            nn.Sigmoid()
        )

        self.out_block = ResidualBlock(64, (54, 54))
        self.attention_downsample = nn.UpsamplingBilinear2d(size=(60, 60))

    def forward(self, x):
        residual_one = self.residual_one(x)
        trunk = self.trunk(residual_one)
        pool_one = self.mpool(residual_one)

        residual_two = self.residual_two(pool_one)
        skip_conn = self.skip_one(residual_two)
        interp_one = self.upsample(residual_two)

        attention_block = self.activation_block(interp_one)
        attention_block = self.attention_downsample(attention_block)

        combination_block = (1 + attention_block) * trunk
        combination_block = combination_block + skip_conn
        out = self.out_block(combination_block)
        return out


