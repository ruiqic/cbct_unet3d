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
        num_stages = len(strides) - 1

        # Encoder
        encoders = [ConvBlock(in_channels, channels[0], prelu, stride=strides[0])]
        for i in range(num_stages):
            encoders.append(ConvBlock(channels[i], channels[i+1], prelu, stride=strides[i+1]))

        # Decoder
        decoders = []
        for i in range(num_stages):
            decoders.append(UpConvBlock(channels[num_stages-i], channels[num_stages-i-1], 
                                             prelu, kernel_size=strides[num_stages-i], stride=strides[num_stages-i]))

        # Final convolutions for deep supervision
        finals = []
        for i in range(num_stages):
            finals.append(nn.Conv3d(channels[num_stages-i-1], num_classes, kernel_size=1, bias=True))
            
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.finals = nn.ModuleList(finals)

    def forward(self, x, deep_supervision=True):
        
        xs = []
        for encoder in self.encoders:
            x = encoder(x)
            xs.append(x)
        # xs = [x1, x2, x3, x4, x5, x6]

        decoded = []
        for decoder, x_stage in zip(self.decoders, reversed(xs[:-1])):
            x = decoder(x, x_stage)
            decoded.append(x)
            
        if deep_supervision:
            outs = []
            for final, dec in zip(self.finals, decoded):
                outs.append(final(dec))
            return outs
        else:
            return self.finals[-1](x)
        