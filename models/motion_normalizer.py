import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import LambdaLR, ResidualBlock, LayerNorm


class MotionNormalizer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, dim=64, encoder_res=3, decoder_res=3, n_sampling=2):
        super(MotionNormalizer, self).__init__()
        
        ## flow encoder
        encoder_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        for _ in range(n_sampling):
            encoder_layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        for _ in range(encoder_res):
            encoder_layers += [ResidualBlock(dim, norm="in")]

        self.encoder = nn.Sequential(*encoder_layers)

        ## flow decoder
        decoder_layers = []
        dim = dim * 2 ** n_sampling
        for _ in range(decoder_res):
            decoder_layers += [ResidualBlock(dim, norm="in")]

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

    def forward(self, flow):
        x = self.encoder(flow)
        x = self.decoder(x)
        return x



class MotionDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(MotionDiscriminator, self).__init__()

        input_shape = (in_channels, 128, 128)
        channels, height, width = input_shape
        ## calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, flow):
        return self.model(flow)



        
class EmotionDiscriminator(nn.Module):
    
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, 8, 3)
        self.batch_norm_0 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool_0 = nn.MaxPool2d(2, 2) 
        
        self.conv_1 = nn.Conv2d(8, 32, 5)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.maxpool_1 = nn.MaxPool2d(2, 2)
        
        self.conv_2 = nn.Conv2d(32, 128, 4)
        self.batch_norm_2 = nn.BatchNorm2d(128)
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        
        self.fc_3 = nn.Linear(128*13*13, 512)
        self.reset_parameters_linear(self.fc_3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.fc_4 = nn.Linear(512, 128)
        self.reset_parameters_linear(self.fc_4)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.fc_5 = nn.Linear(128, n_classes)
        self.reset_parameters_linear(self.fc_5)

    def reset_parameters_linear(self, layer): 
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5)) 
        if layer.bias is not None: 
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight) 
            bound = 1 / np.sqrt(fan_in) 
            nn.init.uniform_(layer.bias, -bound, bound) 
        
    def forward(self, x):
        x = self.maxpool_0(F.relu(self.batch_norm_0(self.conv_0(x))))
        x = self.maxpool_1(F.relu(self.batch_norm_1(self.conv_1(x))))
        x = self.maxpool_2(F.relu(self.batch_norm_2(self.conv_2(x)))) 
        
        #print(x.shape)
        
        x = x.view(-1, 128*13*13)
        x = self.dropout_3(self.fc_3(x))
        x = self.dropout_4(self.fc_4(x))
        x = self.fc_5(x)
     
        return x