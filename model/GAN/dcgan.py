from typing import Any
from torch import nn
import torch
from utils import get_padding_t

class ConvtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_last=False):
        super(ConvtBlock, self).__init__()

        self.is_last = is_last
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh() if is_last else nn.Tanh()

    def forward(self, x):
        x = self.convt(x)
        if not self.is_last:
            x = self.bn(x)
        x = self.activation(x)
        return x
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky_slope, is_bn):
        super(ConvBlock, self).__init__()

        self.layers = []

        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        if is_bn:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if leaky_slope is not None:
            self.layers.append(nn.LeakyReLU(leaky_slope))
        else: 
            self.layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.layers(x)
        return x


class Generator(nn.Module):
    def __init__(self, kernel_sizes, channels, strides, paddings):
        super(Generator, self).__init__()

        self.layers = []

        for i in range(len(strides)):
            is_last = (i == (len(strides) - 1))
            self.layers.append(ConvtBlock(channels[i], channels[i+1], kernel_sizes[i], 
                                         strides[i], paddings[i], is_last))
            
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        x = self.layers(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, leaky_slope):
        super(Discriminator, self).__init__()

        self.layers = []

        for i in range(len(strides)):
            leaky_slope_i = leaky_slope if i != len(strides) - 1 else None
            is_bn = True
            if i == 0 or i == len(strides) - 1:
                is_bn = False
            
            self.layers.append(ConvBlock(channels[i], channels[i+1], kernel_sizes[i], 
                                         strides[i], paddings[i], leaky_slope_i, is_bn))
            
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape((x.shape[0], 1))
        return x
        



            