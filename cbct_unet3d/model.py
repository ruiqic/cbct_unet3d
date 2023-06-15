import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prelu, kernel_size=3, stride=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prelu, kernel_size=2, stride=2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=True)
        self.conv = ConvBlock(2*out_channels, out_channels, prelu, kernel_size=3, stride=1)

    def forward(self, x, x_skip):
        x = self.upconv(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, strides=[1,2,2,2,2,2], channels=[32,64,128,256,320,320], prelu=False):
        super(UNet3D, self).__init__()

        self.num_classes = num_classes

        # Encoder
        self.conv1 = ConvBlock(in_channels, channels[0], prelu, stride=strides[0])
        self.conv2 = ConvBlock(channels[0], channels[1], prelu, stride=strides[1])
        self.conv3 = ConvBlock(channels[1], channels[2], prelu, stride=strides[2])
        self.conv4 = ConvBlock(channels[2], channels[3], prelu, stride=strides[3])
        self.conv5 = ConvBlock(channels[3], channels[4], prelu, stride=strides[4])

        # Bottleneck
        self.bottleneck = ConvBlock(channels[4], channels[5], prelu, stride=strides[5])

        # Decoder
        self.upconv5 = UpConvBlock(channels[5], channels[4], prelu, kernel_size=strides[5], stride=strides[5])
        self.upconv4 = UpConvBlock(channels[4], channels[3], prelu, kernel_size=strides[4], stride=strides[4])
        self.upconv3 = UpConvBlock(channels[3], channels[2], prelu, kernel_size=strides[3], stride=strides[3])
        self.upconv2 = UpConvBlock(channels[2], channels[1], prelu, kernel_size=strides[2], stride=strides[2])
        self.upconv1 = UpConvBlock(channels[1], channels[0], prelu, kernel_size=strides[1], stride=strides[1])

        # Final convolutions for deep supervision
        self.final_conv1 = nn.Conv3d(channels[4], num_classes, kernel_size=1, bias=True)
        self.final_conv2 = nn.Conv3d(channels[3], num_classes, kernel_size=1, bias=True)
        self.final_conv3 = nn.Conv3d(channels[2], num_classes, kernel_size=1, bias=True)
        self.final_conv4 = nn.Conv3d(channels[1], num_classes, kernel_size=1, bias=True)
        self.final_conv5 = nn.Conv3d(channels[0], num_classes, kernel_size=1, bias=True)

    def forward(self, x, deep_supervision=True):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.bottleneck(x5)

        x = self.upconv5(x, x5)
        if deep_supervision:
            out1 = self.final_conv1(x)
        
        x = self.upconv4(x, x4)
        if deep_supervision:
            out2 = self.final_conv2(x)
        
        x = self.upconv3(x, x3)
        if deep_supervision:
            out3 = self.final_conv3(x)
        
        x = self.upconv2(x, x2)
        if deep_supervision:
            out4 = self.final_conv4(x)
        
        x = self.upconv1(x, x1)
        out5 = self.final_conv5(x)

        if deep_supervision:
            return [out1, out2, out3, out4, out5]
        else:
            return out5
        