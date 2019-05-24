import torch
from torch import nn
from torchvision import models

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, input):
        return self.block(input)


class TernausNet(nn.Module):

    def __init__(self, num_filters=64, num_classes=1, pretrained=True):
        super().__init__()
        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.max_pool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.conv2 = self.encoder[3]
        self.conv3_1 = self.encoder[6]
        self.conv3_2 = self.encoder[8]
        self.conv4_1 = self.encoder[11]
        self.conv4_2 = self.encoder[13]
        self.conv5_1 = self.encoder[16]
        self.conv5_2 = self.encoder[18]

        self.deconv_5 = DecoderBlock(num_filters * (2 ** 3), num_filters * (2 ** 3), num_filters * (2 ** 2))
        self.deconv_4 = DecoderBlock(num_filters * (2 ** 3 + 2 ** 2), num_filters * (2 ** 3), num_filters * (2 ** 2))
        self.deconv_3 = DecoderBlock(num_filters * (2 ** 3 + 2 ** 2), num_filters * (2 ** 2), num_filters * 2)
        self.deconv_2 = DecoderBlock(num_filters * (2 * 2 + 2), num_filters * 2, num_filters)
        self.deconv_1 = DecoderBlock(num_filters * (2 + 1), num_filters, num_filters // 2)

        self.final = nn.Conv2d(num_filters // 2 + num_filters, num_classes, 3, padding=1)

    def forward(self, input_image):
        conv1 = self.relu(self.conv1(input_image))  # 64x512x512
        conv2 = self.relu(self.conv2(self.max_pool2d(conv1)))  # 128x256x256
        conv3_2 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.max_pool2d(conv2)))))  # 256x128x128
        conv4_2 = self.relu(self.conv4_2(self.relu(self.conv4_1(self.max_pool2d(conv3_2)))))  # 512x64x64
        conv5_2 = self.relu(self.conv5_2(self.relu(self.conv5_1(self.max_pool2d(conv4_2)))))  # 512x32x32

        out = self.deconv_5(self.max_pool2d(conv5_2))
        out = self.deconv_4(torch.cat([out, conv5_2], 1))
        out = self.deconv_3(torch.cat([out, conv4_2], 1))
        out = self.deconv_2(torch.cat([out, conv3_2], 1))
        out = self.deconv_1(torch.cat([out, conv2], 1))
        out = self.final(torch.cat([out, conv1], 1))

        return out
