import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LambdaLR, ResidualBlock, LayerNorm


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(Encoder, self).__init__()

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


class MotionWarper(torch.nn.Module):
    def __init__(self, motion_channels=2, rgb_channels=3, dim=64, encoder_res=3, decoder_res=3, n_sampling=2):
        super(MotionWarper, self).__init__()
        
        ## encoders
        self.identity_encoder = Encoder(in_channels=3, dim=dim,n_residual=encoder_res, n_downsample=n_sampling)
        self.motion_encoder   = Encoder(in_channels=2, dim=dim,n_residual=encoder_res, n_downsample=n_sampling)

        ## decoder
        decoder_layers = []
        dim = 2 * dim * 2 ** n_sampling
        for _ in range(decoder_res):
            decoder_layers += [ResidualBlock(dim)]

        for _ in range(n_sampling):
            decoder_layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        decoder_layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, image, flow):
        
        identity = self.identity_encoder(image)
        motion   = self.motion_encoder(flow)
        
        x = torch.cat((identity, motion), dim=1)
        x = self.decoder(x)
        
        return x