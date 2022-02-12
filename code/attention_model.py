from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn, TemporalAttn
from basic_layers import BasicBlock, AttentionBasicBlock
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
import pdb

class BaseModel(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv_1 = self._make_layer(3, 64, 3)
        self.res_block_1 = BasicBlock(64)
        self.att_block_1 = AttentionBasicBlock(64)
        self.res_block_2 = BasicBlock(64, 32)
        self.flat_layer = nn.Linear(
            in_features= 506944,
            out_features=n_classes,
            bias=True
        )
        self.n_classes = n_classes

    def _make_layer(self, input_channels, out_features, kernel_size):
        return nn.Sequential(*[
            nn.Conv2d(input_channels, out_features, kernel_size),
            nn.BatchNorm2d(out_features, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
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
        print("1st convolutional layer beginning")
        output_1 = self.conv_1(x)
        print("1st residual layer beginning")
        res_layer = self.res_block_1(output_1)
        print("1st attention layer beginning")
        att_layer = self.att_block_1(res_layer)
        print("2nd residual layer beginning")
        res_layer_2 = self.res_block_2(att_layer)
        print(f"Tensor shape prior to flat layer: {res_layer_2.shape}")
        flattened = res_layer_2.reshape(-1)
        print(flattened.shape)
        final = self.flat_layer(flattened)
        print("Final flat layer finish")
        return final


model= BaseModel(27)
print(summary(model, input_size=(3, 180, 180)))