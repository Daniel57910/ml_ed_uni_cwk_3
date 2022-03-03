from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import BasicBlock, AttentionBasicBlock
from torchvision import models
import torch.nn.functional as F
# from torchsummary import summary
import pdb

class BaseModel(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv_1 = self._make_layer(3, 64, 3)
        self.res_block_1 = BasicBlock(64)
        self.att_block_1 = AttentionBasicBlock(64)
        self.res_block_2 = BasicBlock(64, 32)
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

        self._initialize_weights()
        self.activation = nn.Sigmoid()
        self.n_classes = n_classes

    def _make_layer(self, input_channels, out_features, kernel_size):
        return nn.Sequential(*[
            nn.Conv2d(input_channels, out_features, kernel_size),
            nn.BatchNorm2d(out_features, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2)
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
        res_layer = self.res_block_1(output_1)
        att_layer = self.att_block_1(res_layer)
        res_layer_2 = self.res_block_2(att_layer)
        linear_layer_1 = self.linear_layer_1(res_layer_2)
        linear_layer_2 = self.linear_layer_2(linear_layer_1)
        activation_layer = self.activation_layer(linear_layer_2)
        activation = self.activation(activation_layer)
        """
        Casting required for BCE loss
        """
        return activation.double()


# print(summary(model, (3, 180, 180)))