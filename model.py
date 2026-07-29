"""
U-Net architecture for DDPM on MNIST (28×28 grayscale).

Key design choices — purely convolutional, no attention or Transformer components:
  • Learned MLP time-embedding (simple normalise-and-project approach)
  • Residual blocks with GroupNorm + SiLU + Dropout
  • Down-sampling with strided convolution, up-sampling with nearest + conv
  • Skip connections between encoder and decoder (classic U-Net)

The network predicts the noise ε_θ(x_t, t) added during the forward process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Learned time embedding ────────────────────────────────────────────────────

class LearnedTimeEmbedding(nn.Module):
    """
    Simple learned time embedding using a small MLP.

    Maps integer timestep t ∈ {0, …, T-1} to a dim-dimensional vector:
      1. Normalize  t  →  t / T  ∈ [0, 1]
      2. Project through a 2-layer MLP: Linear(1 → dim) → SiLU → Linear(dim → dim)

    This is a purely learned mapping — no special mathematical formulas,
    just standard neural-network layers that learn to represent time.
    """

    def __init__(self, dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor):
        # Normalize timestep to [0, 1]
        t_norm = t.float().unsqueeze(1) / self.max_timesteps   # (B, 1)
        return self.net(t_norm)  # (B, dim)


# ─── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Two-conv residual block with time-conditioning:

        h = GroupNorm → SiLU → Conv → + time_mlp(t) → GroupNorm → SiLU → Dropout → Conv
        out = h + skip(x)
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection (1×1 conv if channels change)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding (broadcast over spatial dims)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Spatial down-sampling by factor 2 using strided convolution."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial up-sampling by factor 2 using nearest + conv."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ─── Full U-Net ────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net noise-prediction network ε_θ(x_t, t).

    Purely convolutional architecture — no attention, no Transformer components.

    Architecture (for MNIST 28×28, model_channels=128):
      Time embedding: Learned MLP  (normalise t → 4-layer MLP → dim)
      Encoder:
        28×28 → [ResBlock×2 @ 128] → Downsample → 14×14
        14×14 → [ResBlock×2 @ 256] → Downsample → 7×7
      Bottleneck:
        7×7   → [ResBlock×2 @ 512]
      Decoder (mirror of encoder, with skip connections):
        7×7   → Upsample → 14×14 → [ResBlock×3 @ 256]
        14×14 → Upsample → 28×28 → [ResBlock×3 @ 128]
      Output: GroupNorm → SiLU → 1×1 conv → 1 channel
    """

    def __init__(self, in_channels: int = 1, model_channels: int = 64,
                 channel_mult=(1, 2, 4), num_res_blocks: int = 2,
                 time_emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # ── Time embedding MLP ──────────────────────────────────────────────
        # Learned projection: normalise t → [0,1], then 4-layer MLP
        self.time_embed = nn.Sequential(
            LearnedTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Initial projection ──────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        channels = [model_channels]
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                channels.append(ch)

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            ResidualBlock(ch, ch, time_emb_dim, dropout),
        ])

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.decoder_blocks.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch
            if level > 0:
                self.upsamplers.append(Upsample(ch))

        # ── Output ──────────────────────────────────────────────────────────
        self.final_norm = nn.GroupNorm(min(32, ch), ch)
        self.final_conv = nn.Conv2d(ch, in_channels, 1)

        # Store architecture info
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) noisy image x_t
            t: (B,) integer time-steps

        Returns:
            (B, C, H, W) predicted noise ε_θ
        """
        t_emb = self.time_embed(t)       # (B, time_emb_dim)
        x = self.init_conv(x)            # (B, model_ch, H, W)

        # ── Encoder ─────────────────────────────────────────────────────────
        skips = [x]
        block_idx = 0
        down_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                x = self.encoder_blocks[block_idx](x, t_emb)
                skips.append(x)
                block_idx += 1
            if level < len(self.channel_mult) - 1:
                x = self.downsamplers[down_idx](x)
                skips.append(x)
                down_idx += 1

        # ── Bottleneck ──────────────────────────────────────────────────────
        for block in self.bottleneck:
            x = block(x, t_emb)

        # ── Decoder ─────────────────────────────────────────────────────────
        block_idx = 0
        up_idx = 0
        for level in reversed(range(len(self.channel_mult))):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = self.decoder_blocks[block_idx](x, t_emb)
                block_idx += 1
            if level > 0:
                x = self.upsamplers[up_idx](x)
                up_idx += 1

        # ── Output head ────────────────────────────────────────────────────
        x = self.final_conv(F.silu(self.final_norm(x)))
        return x
