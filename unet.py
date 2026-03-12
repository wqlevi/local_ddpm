from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    device = timesteps.device
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Small U-Net backbone for DDPM noise prediction.
    Expects x as BCHW and t as shape [B] (int/float timesteps).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        t_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.t_emb_dim = t_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.down1 = ResBlock(c1, c1, t_emb_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock(c2, c2, t_emb_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResBlock(c3, c3, t_emb_dim)
        self.mid2 = ResBlock(c3, c3, t_emb_dim)

        self.upsample2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock(c2 + c2, c2, t_emb_dim)
        self.upsample1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(c1 + c1, c1, t_emb_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | int | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device)
        if t.ndim == 0:
            t = t[None]
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.repeat(x.shape[0])
        t = t.to(device=x.device)

        t_emb = timestep_embedding(t, self.t_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        d2_in = self.downsample1(d1)
        d2 = self.down2(d2_in, t_emb)
        d3 = self.downsample2(d2)

        m = self.mid2(self.mid1(d3, t_emb), t_emb)

        u2 = self.upsample2(m)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = self.upsample1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, t_emb)

        return self.out_conv(self.out_act(self.out_norm(u1)))
