import numpy as np
import torch.utils.data
from torch.nn import functional as F
import torch


class Block(torch.nn.Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()
        # self.norm = torch.nn.Dropout2d(0.5)
        self.norm = torch.nn.BatchNorm2d(outChannels)
        self.conv1 = torch.nn.Conv2d(inChannels, outChannels, 3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(outChannels, outChannels, 3, 1, 1)

    def forward(self, x):
        return self.relu(self.conv2(self.norm(self.relu(self.conv1(x)))))


class Encoder(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.encblocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        blockoutputs = []

        for block in self.encblocks:
            x = block(x)
            blockoutputs.append(x)
            x = self.pool(x)

        return blockoutputs


class Decoder(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.decblocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            encFeat = encFeatures[i]
            x = self.upsample(x)
            #x = torch.cat([x, encFeat], dim=1)
            x[:, 0:encFeat.size(1), :, :] += encFeat
            x = self.decblocks[i](x)

        return x


class Unet(torch.nn.Module):
    """ Classical Unet implementation.
    See https://arxiv.org/abs/1505.04597 for reference.
    """

    def __init__(self, encChannels = (1, 16, 32, 64, 128, 256), decChannels = (256, 128, 64, 32, 16), nbClasses = 2, outSize = (512, 512)):
        super().__init__()

        self.Encoder = Encoder(encChannels)
        self.Decoder = Decoder(decChannels)

        self.head = torch.nn.Conv2d(decChannels[-1], nbClasses, 1, 1)
        self.outsize = outSize

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x[::-1][0], x[::-1][1:])
        x = self.head(x)
        return x