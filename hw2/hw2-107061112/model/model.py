import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Autoencoder(BaseModel):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # Interpolate(size=(13,13), scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            Interpolate(size=(26,26), scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
        )
    def forward(self, inputs, decode_only=False):
        if decode_only:
            decoded = self.decoder(inputs)
            return decoded
        latent_vec = self.encoder(inputs)
        decoded = self.decoder(latent_vec)
        return latent_vec, decoded

class Interpolate(nn.Module):
    def __init__(self, size, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
