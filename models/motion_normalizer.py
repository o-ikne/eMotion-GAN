import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import LambdaLR, ResidualBlock, LayerNorm

class MotionEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(MotionEncoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


    
class MotionDecoder(nn.Module):
    def __init__(self, out_channels=2, dim=64, n_residual=3, n_upsample=2):
        super(MotionDecoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MotionNormalizer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, dim=64, encoder_res=3, decoder_res=3, n_sampling=2):
        super(MotionNormalizer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.encoder_res = encoder_res
        self.decoder_res = decoder_res
        self.n_sampling = n_sampling

        ## flow encoder
        self.encoder = MotionEncoder(in_channels=in_channels,
                                     dim=dim,
                                     n_residual=encoder_res,
                                     n_downsample=n_sampling)
        
        ## flow decoder
        self.decoder = MotionDecoder(out_channels=out_channels,
                                     dim=dim,
                                     n_residual=decoder_res,
                                     n_upsample=n_sampling)


    def forward(self, flow):
        x = self.encoder(flow)
        x = self.decoder(x)
        return x
