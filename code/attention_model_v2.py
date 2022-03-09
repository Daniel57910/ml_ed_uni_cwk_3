from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers_v2 import make_convolution_layer, ResidualBlock, AttentionBlock1
from torchvision import models
import torch.nn.functional as F
# from torchsmmary import summary
import pdb

class AttModelV2(nn.Module):
    def __init__(self, n_classes) -> None:
        super(AttModelV2, self).__init__()
        self.n_classes = n_classes
        self.conv_1 = make_convolution_layer(3, 64, pool_type="max", dropout_rate=0.2)
        self.residual_one = ResidualBlock(64, (82, 82))
        self.attention_one = AttentionBlock1(64, 64, 0.4)
        self.residual_two = ResidualBlock(64, upsample_size=(48, 48), channel_output=32)
        self.mpool = nn.MaxPool2d(3)

        self.linear_layer_1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=8192,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout()
        )

        self.linear_layer_2 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.ReLU(),
            nn.Dropout()
        )
        self.activation_layer = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=n_classes
            )
        )


    def forward(self, x):
        out = self.conv_1(x)

        res_1 = self.residual_one(out)
        att_1 = self.attention_one(res_1)
        res_2 = self.residual_two(att_1)
        mpool = self.mpool(res_2)
        linear_one = self.linear_layer_1(mpool)
        linear_two = self.linear_layer_2(linear_one)
        activation = self.activation_layer(linear_two)
        return activation


