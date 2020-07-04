import torch.nn as nn


# CONVOLUTION BLOCK 1
def Conv2d_BN(inChannels, outChannels, kernel=3, dropout=0.1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(outChannels),
        nn.Dropout(dropout)
    )  # output_size = 32, Ni = 30; rf = 1


# TRANSITION BLOCK 1
def Conv2d_BN(inChannels, outChannels, kernel=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), padding=0, bias=False),
    )  # output_size = 30; rf = 1


def Maxpooling(kernel):
    return nn.MaxPool2d(kernel, kernel) # output_size = 15; rf = 2


# CONVOLUTION BLOCK 2
def Conv2d_BN(inChannels, outChannels, kernel=3, dropout=0.1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(outChannels),
        nn.Dropout(dropout)
    )  # output_size = 13; rf = 6


# TRANSITION BLOCK 2
def Conv2d_BN(inChannels, outChannels, kernel=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), padding=0, bias=False),
    )  # output_size = 13; rf = 6


def Maxpooling(kernel):
    return nn.MaxPool2d(kernel, kernel) # output_size = 7; rf = 8


# CONVOLUTION BLOCK 3
# Depthwise Separable Convolution
def DepthwiseConv2d(inChannels, outChannels, kernel=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(3, 3), padding=0,
                  bias=False), )  # output_size = 5; rf = 16


def PointwiseConv2d(inChannels, outChannels, kernel=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(outChannels),
    )  # output_size = 5; rf = 16
def Maxpooling(kernel):
    return nn.MaxPool2d(kernel, kernel) # output_size = 5; rf = 20

# CONVOLUTION BLOCK 4
# Dilated Convolution
def DilationConv2d(inChannels, outChannels, kernel=3, padding=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(3, 3), padding=1, dilation
        =1, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(outChannels),
    )  # output_size = 1; rf = 52


# OUTPUT BLOCK
def gap(kernel):
    return nn.Sequential(nn.AvgPool2d(kernel_size=kernel)
                         ) # output_size = 1; rf = 52


def Conv1d(inChannels, outChannels, kernel=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), padding=0, bias=False),
    ) # output_size = 1; rf = 52






