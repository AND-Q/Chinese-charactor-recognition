import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
import time
import os

def vgg_block(num_convs, in_channels, out_channels):#vgg块
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

