from typing import Callable
import torch.nn.init as init
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# Inspired by https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class DiceLoss(nn.Module):
    """
    DiceLoss is a custom loss function used for evaluating the similarity between the predicted output and the ground truth in segmentation tasks.
    It is based on the Dice coefficient, which we saw in the lecture.

    Attributes:
        smooth (float): A smoothing factor to avoid division by zero errors.

    Methods:
        forward(model_out, target): Compute the loss between the model output and the target.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, model_out, target):
        """
        Compute the loss between the model output and the target.
        This function first applies a softmax activation to the model output, then
        reshapes both the model output and the target to be 2D tensors. It calculates
        the intersection and denominator for the Dice coefficient and computes the
        loss as 1 minus the Dice coefficient.
        Args:
            model_out (torch.Tensor): The output from the model, expected to have the same shape as the target.
            target (torch.Tensor): The ground truth tensor, expected to have the same shape as the model output.
        Returns:
            torch.Tensor: The mean loss value.
        """
        assert (
            model_out.size() == target.size()
        ), f"'output' and 'target' must have the same shape ({model_out.size()} != {target.size()})"
        model_out = torch.softmax(model_out, dim=1)

        # Flatten the tensors to 2D arrays for easier computation
        model_out = model_out.contiguous().view(model_out.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Calculate the intersection and denominator parts for the Dice coefficient
        intersect = torch.sum(torch.mul(model_out, target), dim=1) + self.smooth
        den = torch.sum(model_out.pow(2) + target.pow(2), dim=1) + self.smooth

        # Calculate the Dice coefficient as seen in the lecture and return 1 minus the coefficient as the loss
        loss = 1 - (2.0 * (intersect / den))

        return loss.mean()


class _EncoderBlock(nn.Module):
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
        dec4 = self.dec4(
            torch.cat([center, F.upsample(enc4, center.size()[2:], mode="bilinear")], 1)
        )
        dec3 = self.dec3(
            torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode="bilinear")], 1)
        )
        dec2 = self.dec2(
            torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode="bilinear")], 1)
        )
        dec1 = self.dec1(
            torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode="bilinear")], 1)
        )
        final = self.final(dec1)

        return F.upsample(final, x.size()[2:], mode="bilinear")
