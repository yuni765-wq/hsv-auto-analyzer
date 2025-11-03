# -*- coding: utf-8 -*-
"""
2D Conv-VAE for HSV/GSP frames (128x128, grayscale)
- Latent dim default = 16
- Compatible with torch >= 1.12
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
z_dim: int = 16


class Encoder2D(nn.Module):
def __init__(self, z_dim=16):
super().__init__()
self.conv = nn.Sequential(
nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
)
self.fc_mu = nn.Linear(128*16*16, z_dim)
self.fc_logvar = nn.Linear(128*16*16, z_dim)


def forward(self, x):
h = self.conv(x)
h = h.view(x.size(0), -1)
mu, logvar = self.fc_mu(h), self.fc_logvar(h)
return mu, logvar


class Decoder2D(nn.Module):
def __init__(self, z_dim=16):
super().__init__()
self.fc = nn.Linear(z_dim, 128*16*16)
self.deconv = nn.Sequential(
nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
)


def forward(self, z):
h = self.fc(z).view(z.size(0), 128, 16, 16)
return self.deconv(h)


class VAE2D(nn.Module):
def __init__(self, cfg: VAEConfig = VAEConfig()):
super().__init__()
self.z_dim = cfg.z_dim
self.enc = Encoder2D(cfg.z_dim)
self.dec = Decoder2D(cfg.z_dim)


@staticmethod
def reparam(mu, logvar):
std = (0.5 * logvar).exp()
eps = torch.randn_like(std)
return mu + eps * std


def forward(self, x):
mu, logvar = self.enc(x)
z = self.reparam(mu, logvar)
x_hat = self.dec(z)
return x_hat, mu, logvar


def kl_divergence(mu, logvar):
return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def recon_loss(x, x_hat):
return F.l1_loss(x_hat, x)