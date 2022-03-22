from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers_v1 import BasicBlock, AttentionBasicBlock
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
import pdb

class AttModelV1(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv_1 = self._make_layer(3, 64, 3)

        self.res_block_1 = BasicBlock(64)
        self.att_block_1 = AttentionBasicBlock(64, 249)

        self.res_block_2 = BasicBlock(64, 64)
        self.att_block_2 = AttentionBasicBlock(64, 249)

        self.res_block_3 = BasicBlock(64, 64)

        self.av_pool_layer = nn.AvgPool2d(kernel_size=3)

        self.linear_layer_1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=440896,
                out_features=256
            ),
            nn.Dropout(),
            nn.ReLU()
        )

        self.linear_layer_2 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.activation_layer = nn.Linear(
                in_features=128,
                out_features=n_classes
            )

        self._initialize_weights()
        self.n_classes = n_classes

    def _make_layer(self, input_channels, out_features, kernel_size):
        return nn.Sequential(*[
            nn.Conv2d(input_channels, out_features, kernel_size),
            nn.BatchNorm2d(out_features, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        ])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output_1 = self.conv_1(x)

        res_layer_1 = self.res_block_1(output_1)
        att_layer_1 = self.att_block_1(res_layer_1)

        res_layer_2 = self.res_block_2(att_layer_1)
        att_layer_2 = self.att_block_2(res_layer_2)

        res_block_3 = self.res_block_3(att_layer_2)

        av_pool_layer = self.av_pool_layer(res_block_3)

        linear_layer_1 = self.linear_layer_1(av_pool_layer)
        linear_layer_2 = self.linear_layer_2(linear_layer_1)

        activation_layer = self.activation_layer(linear_layer_2)
        return activation_layer

