from torch import nn
import torch
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        # for encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(self.h_dim, self.z_dim)
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.h_dim)
        
        # for decoder
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(self.h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        mu = self.fc1(h1)
        logvar = self.fc2(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 0.5 for square root (variance to standard deviation)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar