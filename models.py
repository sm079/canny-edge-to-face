import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)

class DepthToSpace(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.size = size

    def forward(self, x):
        b, c, h, w = x.shape
        oh, ow = h * self.size, w * self.size
        oc = c // (self.size * self.size)
        x = x.view(b, self.size, self.size, oc, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.contiguous().view(b, oc, oh, ow)
        return x
    
class OutConv(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        self.out_conv = nn.ModuleList([
            nn.Conv2d(n_ch, 3, kernel_size=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1),
        ])

    def forward(self, x):
        return torch.cat([i(x) for i in self.out_conv], dim=1)
    
    
class Res(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1),
        )

        self.fuse = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.fuse(x + self.conv(x))

class CannyEncoder(nn.Module):
    def __init__(self, e_ch):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(     3, e_ch*1, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True), Res(e_ch*1),
            nn.Conv2d(e_ch*1, e_ch*2, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(e_ch*2, e_ch*4, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(e_ch*4, e_ch*8, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.1, inplace=True), Res(e_ch*8),
            nn.Flatten(), PixelNorm(), nn.Linear(512*8*8, 128), nn.Linear(128, 512*8*8), nn.Unflatten(1, (512, 8, 8))
        )

    def forward(self, x):
        return self.image_encoder(x)
    
class CannyDecoder(nn.Module):
    def __init__(self, d_ch):
        super().__init__()

        self.image_decoder = nn.Sequential(
            nn.Conv2d(   512, d_ch*8*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*8),
            nn.Conv2d(d_ch*8, d_ch*4*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*4),
            nn.Conv2d(d_ch*4, d_ch*2*4, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True), DepthToSpace(2), Res(d_ch*2),
            OutConv(d_ch*2), DepthToSpace(2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.image_decoder(x)