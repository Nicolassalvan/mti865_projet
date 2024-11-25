import torch.nn.init as init
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    """
    Initialize the weights of the given models using Kaiming normal initialization for Conv2d and Linear layers,
    and setting BatchNorm2d weights to 1 and biases to 0.

    Args:
        *models: One or more PyTorch models whose weights need to be initialized.
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    """
    A helper class that defines the encoder block of the U-Net architecture, which consists of two 3x3 convolutions
    with batch normalization and ReLU activation, followed by a 2x2 max pooling operation.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        dropout (bool): Whether to apply dropout between the two convolutions.

    Attributes:
        encode (nn.Sequential): The encoder block.
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    A helper class that defines the decoder block of the U-Net architecture, which consists of a 3x3 convolution 
    with batch normalization and ReLU activation, followed by a 2x2 transposed convolution.

    Args:
        in_channels (int): The number of input channels.
        middle_channels (int): The number of output channels for the convolution.
        out_channels (int): The number of output channels.

    Attributes:
        decode (nn.Sequential): The decoder block.

    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    """
    The U-Net architecture for image segmentation. 

    Args:
        num_classes (int): The number of classes to segment.

    Attributes:
        enc1 (nn.Module): The first encoder block.
        enc2 (nn.Module): The second encoder block.
        enc3 (nn.Module): The third encoder block.
        enc4 (nn.Module): The fourth encoder block.
        center (nn.Module): The center block.
        dec4 (nn.Module): The fourth decoder block.
        dec3 (nn.Module): The third decoder block.
        dec2 (nn.Module): The second decoder block.
        dec1 (nn.Module): The first decoder block.
        final (nn.Conv2d): The final convolution layer.

    
    """
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(1, 4)
        self.enc2 = _EncoderBlock(4, 8)
        self.enc3 = _EncoderBlock(8, 16)
        self.enc4 = _EncoderBlock(16, 32, dropout=True)
        self.center = _DecoderBlock(32, 64, 32)
        self.dec4 = _DecoderBlock(64, 32, 16)
        self.dec3 = _DecoderBlock(32, 16, 8)
        self.dec2 = _DecoderBlock(16, 8, 4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(4, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)

        return F.upsample(final, x.size()[2:], mode='bilinear')

    