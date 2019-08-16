import torch
from torch import nn
import torch.functional as F

class VNet(nn.Module):

    # for tests

    def __init__(self, num_filters=64, num_classes=3, pretrained=False):
        super().__init__()

        self.conv1 = nn.Conv3d(1, num_classes, kernel_size=3, padding=1)

    def forward(self, image):

        out = self.conv1(image)

        return out