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

    def __init__(self, channel_input, channel_output=None, dropout_rate=0.4):

        if not channel_output:
            channel_output = channel_input

        super(ResidualBlock, self).__init__()
        self.conv_block1 = make_convolution_layer(channel_input, channel_output)
        self.conv_block2 = make_convolution_layer(channel_output, channel_output)
        self.conv_block3 = make_convolution_layer(channel_output, channel_output)
        if channel_output:
            self.conv_upsample = nn.Sequential(
                nn.Conv2d(
                    channel_input,
                    channel_output,
                    kernel_size=3,
                    bias=False),
                nn.UpsamplingNearest2d(
                    size=(82, 82)
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
