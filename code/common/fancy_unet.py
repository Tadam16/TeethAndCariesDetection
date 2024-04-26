import torch.utils.data
import torchvision.transforms
import torch


class StartBlock(torch.nn.Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(outChannels)
        self.conv1 = torch.nn.Conv2d(inChannels, outChannels, 7, 1, 'same')
        self.conv2 = torch.nn.Conv2d(outChannels, outChannels, 5, 1, 'same')

    def forward(self, x):
        return self.norm(torch.relu(self.conv2(torch.relu(self.conv1(x)))))


class DenseBlock(torch.nn.Module):

    def __init__(self, in_channels, conv_channels):
        super(DenseBlock, self).__init__()
        self.norm = torch.nn.BatchNorm2d(conv_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, conv_channels, 1, 1, 'same')
        self.conv2 = torch.nn.Conv2d(conv_channels, conv_channels, 3, 1, 'same')
        self.conv3 = torch.nn.Conv2d(conv_channels, conv_channels, 3, 1, 'same')

    def forward(self, x):
        return self.norm(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x)))))))


class HeadBlock(torch.nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels):
        super(HeadBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, inter_channels, 1, 1, 'same')
        self.conv2 = torch.nn.Conv2d(inter_channels, inter_channels, 3, 1, 'same')
        self.conv3 = torch.nn.Conv2d(inter_channels, out_channels, 3, 1, 'same')

    def forward(self, x):
        return self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x)))))


class Unet(torch.nn.Module):
    """ Custom UNet implementation, with DenseBlocks to increase connectivity
    Similar idea to https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2019.1527
    """

    def __init__(self, in_channels=1, inter_channels=32, height=4, width=2, class_num=3):
        super().__init__()
        self.trans_d = torchvision.transforms.Resize((320, 320), antialias=True)
        self.trans_u = torch.nn.UpsamplingBilinear2d((384, 384))
        self.height = height
        self.width = width
        self.start = StartBlock(in_channels, inter_channels)
        channel_num = inter_channels
        skip_channels = []
        self.enc_blocks = torch.nn.ModuleList()
        self.inter_channels = inter_channels

        for i in range(height + 1):
            for j in range(width):
                self.enc_blocks.append(DenseBlock(channel_num, inter_channels))
                channel_num += inter_channels
            skip_channels.append(channel_num)

        self.dec_blocks = torch.nn.ModuleList()
        skip_channels = skip_channels[::-1]
        up_channels = inter_channels * width
        for i in range(height):
            skip_num = skip_channels[i + 1]
            for j in range(width):
                self.dec_blocks.append(DenseBlock(skip_num + up_channels + j * inter_channels, inter_channels))
            up_channels += width * inter_channels

        head_channels = up_channels + skip_num
        self.head = HeadBlock(head_channels, inter_channels, class_num)

        self.pool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.trans_d(x)
        x = self.start(x)
        skips = []
        for i in range(self.height + 1):
            for j in range(self.width):
                x = torch.cat((x, self.enc_blocks[i * self.width + j](x)), 1)
            skips.append(x)
            if i != self.height:
                x = self.pool(x)

        skips = skips[::-1]
        for i in range(self.height):
            x = x[:, -self.inter_channels * self.width * (i + 1):, :, :]
            x = torch.cat((self.upsample(x), skips[i + 1]), 1)
            for j in range(self.width):
                x = torch.cat((x, self.dec_blocks[i * self.width + j](x)), 1)

        return self.trans_u(self.head(x))
        # return self.head(x)
