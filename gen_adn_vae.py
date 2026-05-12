import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        proj_query = self.query(x).view(batch, -1, h*w).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, h*w)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch, -1, h*w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return self.gamma * out.view(batch, c, h, w) + x

class AdaIN(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.fc = nn.Linear(latent_dim, channels * 2)
    def forward(self, x, style):
        params = self.fc(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = params.chunk(2, 1)
        return (1 + gamma) * self.norm(x) + beta

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, norm=True):
        super().__init__()
        layers = [nn.ReflectionPad2d(padding), nn.Conv2d(in_ch, out_ch, kernel, stride, 0)]
        if norm: layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(2, 64, 7, 1, 3), 
            ConvBlock(64, 128, 4, 2, 1),
            ConvBlock(128, 256, 4, 2, 1),
            SelfAttention(256)
        )
    def forward(self, x, g):
        return self.model(torch.cat([x, g], dim=1))

class ArtifactVAEEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class ArtifactGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.adain = AdaIN(256, latent_dim)
        self.attn = SelfAttention(256)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 128)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64)
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, 7, 1, 0),
            nn.Tanh()
        )
    def forward(self, c, z):
        x = self.adain(c, z)
        x = self.attn(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)

class GenADN_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.EC = ContentEncoder()
        self.EA = ArtifactVAEEncoder()
        self.GA = ArtifactGenerator()
        self.DA = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 4, 1, 1)
        )