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
    dropout_rate=0.5):
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



    def __init__(self, channel_input, channel_output=None, dropout_rate=0.5):

        if not channel_output:
            channel_output = channel_input

        super(BasicBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, 3, padding=1),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_output, channel_output, 3, padding=1),
            nn.BatchNorm2d(channel_output),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        if channel_output:
            self.residual_block = nn.Sequential(
                nn.Conv2d(channel_input, channel_output, 3, padding=1),
                nn.BatchNorm2d(channel_output),
                nn.Dropout(p=0.3)
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
    def __init__(self, channel_num) -> None:
        super(AttentionBasicBlock, self).__init__()
        self.residual_one = BasicBlock(64)
        self.trunk = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64)
        )

        self.softmax = nn.Softmax(dim=0)
        self.upsample = nn.UpsamplingBilinear2d(size=(89, 89))

    def forward(self, x):
        out = self.residual_one(x)
        trunk = self.trunk(x)
        out_pool = nn.MaxPool2d(kernel_size=3, stride=2)(out)
        softmax = self.softmax(out_pool)
        softmax = nn.Dropout(p=0.3)(softmax)
        upsample = self.upsample(softmax)
        trunk_combine = (1 + upsample) * trunk
        return trunk_combine
