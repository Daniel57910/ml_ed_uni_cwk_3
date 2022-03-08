from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers_v2 import make_convolution_layer, ResidualBlock
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
import pdb

class AttModelV2(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.conv_1 = make_convolution_layer(3, 64, pool_type="max", dropout_rate=0.2)
        self.residual_one = ResidualBlock(64, 64)

        self.linear_layer_1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=253472,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3)
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
        return res_1



