from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


# ---------- Shared utils ----------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation helper: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (GLIDE-style)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RectPatchEmbed(nn.Module):
    """Patch embedding for non-square inputs."""

    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"Expected spatial {(self.img_size)}, got {x.shape[-2:]}")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ---------- ViT ----------
@dataclass
class ViTConfig:
    """Config for rectangular 3x24x24 inputs (channels, height, width)."""

    img_size: Tuple[int, int] = (24, 24)
    patch_size: Tuple[int, int] = (3, 3)
    in_chans: int = 3
    embed_dim: int = 128
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    num_outputs: Optional[int] = None  # set for regression/classification heads


class ViTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class C2DBVisionTransformer(nn.Module):
    """
    小型 ViT，适配 3x24x24 网格，默认返回全局池化特征或可选线性头输出。
    """

    def __init__(self, cfg: ViTConfig = ViTConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = RectPatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop_rate, cfg.attn_drop_rate)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_outputs) if cfg.num_outputs is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        if self.head is not None:
            x = self.head(x)
        return x


# ---------- JiT diffusion ----------
@dataclass
class JiTC2DBConfig:
    """Config tuned for torus-encoded C2DB tensors (channels, H, W)."""

    img_size: Tuple[int, int] = (24, 24)
    patch_size: Tuple[int, int] = (3, 3)  # 8x8 patches for 24x24
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_embed_dim: int = 256
    class_embed_dim: int = 256
    num_classes: Optional[int] = None  # set to an int to enable class conditioning


class JiTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = JiTMLP(dim, mlp_ratio, dropout)

        # AdaLN modulation with gates for MSA / MLP, mirroring JiT
        self.mod_linear = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod_linear(cond).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """Project tokens back to patch pixels."""

    def __init__(self, dim: int, patch_size: Tuple[int, int], out_chans: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, patch_size[0] * patch_size[1] * out_chans)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_mod(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class C2DBJiT(nn.Module):
    """
    小型 JiT 风格扩散模型，适配 3x24x24 输入；包含时间/可选标签调制，输出与输入同形状。
    """

    def __init__(self, cfg: JiTC2DBConfig = JiTC2DBConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = RectPatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        if cfg.num_classes is not None:
            self.class_embed = nn.Embedding(cfg.num_classes, cfg.class_embed_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(cfg.class_embed_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
        else:
            self.class_embed = None
            self.class_mlp = None

        self.blocks = nn.ModuleList(
            [
                JiTBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.final = FinalLayer(cfg.embed_dim, cfg.patch_size, cfg.in_chans)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, patch_area * C) -> (B, C, H, W)
        """
        b, num_patches, dim = x.shape
        h_grid, w_grid = self.patch_embed.grid_size
        p_h, p_w = self.cfg.patch_size
        x = x.view(b, h_grid, w_grid, p_h, p_w, self.cfg.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, self.cfg.in_chans, h_grid * p_h, w_grid * p_w)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 24, 24) noisy input.
            timesteps: (B,) diffusion steps.
            labels: Optional (B,) class labels for conditional runs.
        Returns:
            Predicted noise with shape (B, 3, 24, 24).
        """
        t_emb = sinusoidal_timestep_embedding(timesteps, self.cfg.time_embed_dim)
        cond = self.time_mlp(t_emb)

        if self.class_embed is not None and labels is not None:
            y_emb = self.class_embed(labels)
            cond = cond + self.class_mlp(y_emb)

        tokens = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        tokens = self.final(tokens, cond)
        return self.unpatchify(tokens)


__all__ = [
    "ViTConfig",
    "C2DBVisionTransformer",
    "JiTC2DBConfig",
    "C2DBJiT",
    "sinusoidal_timestep_embedding",
    "RectPatchEmbed",
    "modulate",
]
