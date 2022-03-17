import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import pdb

class BasicBlock(nn.Module):
    def __init__(self, channel_input, channel_output=None, dropout_rate=0.5):

        if not channel_output:
            channel_output = channel_input

        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, 3, padding=1),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_output, channel_output, 3, padding=1),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        if channel_output:
            self.residual_block = nn.Sequential(
                nn.Conv2d(channel_input, channel_output, 3, padding=1),
                nn.BatchNorm2d(channel_output),
                nn.Dropout(p=0.2)
            )

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        if out.shape != residual.shape:
            residual = self.residual_block(residual)
            out = out + residual
        return out


class AttentionBasicBlock(nn.Module):
    def __init__(self, channel_num, upsample_size) -> None:
        super(AttentionBasicBlock, self).__init__()
        self.residual_one = BasicBlock(channel_num)
        self.trunk = nn.Sequential(
            BasicBlock(channel_num),
            BasicBlock(channel_num)
        )

        self.sigmoid = nn.Sigmoid()
        self.upsample_out = nn.UpsamplingBilinear2d(size=(upsample_size, upsample_size))
        self.upsample_trunk = nn.UpsamplingBilinear2d(size=(upsample_size, upsample_size))

    def forward(self, x):
        out = self.residual_one(x)
        trunk = self.trunk(x)
        out_pool = nn.MaxPool2d(kernel_size=3, stride=2)(out)
        softmax = self.sigmoid(out_pool)
        upsample = self.upsample_out(softmax)
        if trunk.shape != upsample.shape:
            trunk = self.upsample_trunk(trunk)
        trunk_combine = (1 + upsample) * trunk
        return trunk_combine