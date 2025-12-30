from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from common.crystal import build_knn, cholesky6_to_lattice, frac_mic_dist, gram6_to_lattice, rbf_expand
from data.torus import torus_encode


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@dataclass
class AtomTransformerConfig:
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_embed_dim: int = 256
    num_elements: int = 118
    z_embed_dim: int = 128
    f_embed_dim: int = 128
    rbf_dim: int = 32
    rbf_r_max: float = 6.0
    k_neighbors: int = 32
    fourier_freqs: Tuple[int, ...] = (1, 2, 4, 8)
    pair_mlp_hidden: int = 128
    g_scale: float = 1.0
    cell_bias: bool = True
    cache_neighbors: bool = True
    cache_thresh: float = 1e-3
    cell_rep: str = "gram6"  # gram6 | cholesky6
    chol_log_min: Optional[float] = None
    chol_log_max: Optional[float] = None


class MLP(nn.Module):
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


class GatherAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        bias_nbr: torch.Tensor,
        bias_atom_cell: Optional[torch.Tensor],
        bias_cell_atom: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        n_atoms = seq_len - 1
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_cell = q[:, :, :1, :]
        k_cell = k[:, :, :1, :]
        v_cell = v[:, :, :1, :]
        q_atoms = q[:, :, 1:, :]
        k_atoms = k[:, :, 1:, :]
        v_atoms = v[:, :, 1:, :]

        k = nbr_idx.shape[-1]
        idx = nbr_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, -1, self.head_dim)
        k_atoms_exp = k_atoms.unsqueeze(3).expand(-1, -1, -1, k, -1)
        v_atoms_exp = v_atoms.unsqueeze(3).expand(-1, -1, -1, k, -1)
        k_nbr = torch.gather(k_atoms_exp, 2, idx)
        v_nbr = torch.gather(v_atoms_exp, 2, idx)

        logits_nbr = (q_atoms.unsqueeze(3) * k_nbr).sum(-1) * self.scale
        logits_nbr = logits_nbr + bias_nbr

        logits_cell = (q_atoms * k_cell).sum(-1, keepdim=True) * self.scale
        if bias_atom_cell is not None:
            logits_cell = logits_cell + bias_atom_cell

        nbr_mask = nbr_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        logits_nbr = logits_nbr.masked_fill(~nbr_mask, float("-inf"))

        query_mask = atom_mask > 0.5
        logits_cell = logits_cell.masked_fill(~query_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        logits_nbr = logits_nbr.masked_fill(~query_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        logits = torch.cat([logits_cell, logits_nbr], dim=-1)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        attn_cell = attn[:, :, :, :1]
        attn_nbr = attn[:, :, :, 1:]
        out_atoms = attn_cell * v_cell + torch.sum(attn_nbr.unsqueeze(-1) * v_nbr, dim=-2)

        logits_cell_self = (q_cell * k_cell).sum(-1, keepdim=True) * self.scale
        logits_cell_atoms = torch.matmul(q_cell, k_atoms.transpose(-2, -1)) * self.scale
        if bias_cell_atom is not None:
            logits_cell_atoms = logits_cell_atoms + bias_cell_atom
        logits_cell_atoms = logits_cell_atoms.masked_fill(~query_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        logits_cell_all = torch.cat([logits_cell_self, logits_cell_atoms], dim=-1)
        attn_cell_all = torch.softmax(logits_cell_all, dim=-1)
        attn_cell_all = self.dropout(attn_cell_all)
        attn_cell_self = attn_cell_all[:, :, :, :1]
        attn_cell_atoms = attn_cell_all[:, :, :, 1:]
        out_cell = attn_cell_self * v_cell + torch.matmul(attn_cell_atoms, v_atoms)

        out = torch.cat([out_cell, out_atoms], dim=2)
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.proj(out)


class AtomBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GatherAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        bias_nbr: torch.Tensor,
        bias_atom_cell: Optional[torch.Tensor],
        bias_cell_atom: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            h, nbr_idx, nbr_mask, atom_mask, bias_nbr, bias_atom_cell, bias_cell_atom
        )
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class AtomTransformer(nn.Module):
    def __init__(self, cfg: AtomTransformerConfig = AtomTransformerConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.num_elements + 1
        self._cache: dict[str, torch.Tensor] = {}
        self.z_embed = nn.Embedding(cfg.num_elements + 2, cfg.z_embed_dim, padding_idx=0)
        self.f_proj = nn.Linear(3 * 2 * len(cfg.fourier_freqs), cfg.f_embed_dim)
        self.in_proj = nn.Linear(cfg.z_embed_dim + cfg.f_embed_dim, cfg.embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(6, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        self.blocks = nn.ModuleList(
            [AtomBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.head_f = nn.Linear(cfg.embed_dim, 3)
        self.head_z = nn.Linear(cfg.embed_dim, cfg.num_elements + 1)
        self.head_g = nn.Linear(cfg.embed_dim, 6)

        pair_in = cfg.rbf_dim + 2 * cfg.z_embed_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, cfg.pair_mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.pair_mlp_hidden, cfg.num_heads),
        )
        if cfg.cell_bias:
            bias_in = 3 * cfg.embed_dim
            self.bias_atom_cell = nn.Sequential(
                nn.Linear(bias_in, cfg.pair_mlp_hidden),
                nn.SiLU(),
                nn.Linear(cfg.pair_mlp_hidden, cfg.num_heads),
            )
            self.bias_cell_atom = nn.Sequential(
                nn.Linear(bias_in, cfg.pair_mlp_hidden),
                nn.SiLU(),
                nn.Linear(cfg.pair_mlp_hidden, cfg.num_heads),
            )
        else:
            self.bias_atom_cell = None
            self.bias_cell_atom = None

    def _pair_bias(self, rbf: torch.Tensor, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        bsz, n, k, _ = rbf.shape
        feat = torch.cat([rbf, z_i, z_j], dim=-1)
        bias = self.pair_mlp(feat)
        return bias.permute(0, 3, 1, 2)

    def forward(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        g: torch.Tensor,
        atom_mask: torch.Tensor,
        timesteps: torch.Tensor,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n, _ = frac.shape
        t_emb = sinusoidal_timestep_embedding(timesteps, self.cfg.time_embed_dim)
        cond = self.time_mlp(t_emb)

        z_emb = self.z_embed(z)
        f_emb = self.f_proj(torus_encode(frac, self.cfg.fourier_freqs))
        atom_tokens = self.in_proj(torch.cat([z_emb, f_emb], dim=-1))

        cell_token = self.cell_mlp(g).unsqueeze(1)
        tokens = torch.cat([cell_token, atom_tokens], dim=1)

        use_cache = self.cfg.cache_neighbors and not self.training
        if use_cache and cache_every is not None:
            if step is None or step % max(cache_every, 1) == 0:
                use_cache = False
            else:
                use_cache = all(key in self._cache for key in ("nbr_idx", "nbr_mask", "dist_nbr"))
        if use_cache and cache_every is None:
            if "frac" in self._cache and "g" in self._cache:
                prev_frac = self._cache["frac"]
                prev_g = self._cache["g"]
                if torch.mean((prev_frac - frac) ** 2) < self.cfg.cache_thresh and torch.mean((prev_g - g) ** 2) < self.cfg.cache_thresh:
                    use_cache = True
                else:
                    use_cache = False
            else:
                use_cache = False
        if not use_cache:
            if self.cfg.cell_rep == "cholesky6":
                lattice = cholesky6_to_lattice(g, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max)
                lattice = lattice * self.cfg.g_scale ** 0.5
            else:
                lattice = gram6_to_lattice(g * self.cfg.g_scale)
            dist = frac_mic_dist(frac, lattice, atom_mask)
            nbr_idx, nbr_mask = build_knn(dist, self.cfg.k_neighbors)
            dist_nbr = torch.gather(dist, 2, nbr_idx)
            if self.cfg.cache_neighbors:
                self._cache["frac"] = frac.detach()
                self._cache["g"] = g.detach()
                self._cache["nbr_idx"] = nbr_idx.detach()
                self._cache["nbr_mask"] = nbr_mask.detach()
                self._cache["dist_nbr"] = dist_nbr.detach()
        else:
            nbr_idx = self._cache["nbr_idx"]
            nbr_mask = self._cache["nbr_mask"]
            dist_nbr = self._cache["dist_nbr"]
        rbf = rbf_expand(dist_nbr, self.cfg.rbf_dim, self.cfg.rbf_r_max)
        z_i = z_emb.unsqueeze(2).expand(-1, -1, nbr_idx.shape[-1], -1)
        z_j = z_emb.unsqueeze(2).expand(-1, -1, nbr_idx.shape[-1], -1)
        z_j = torch.gather(z_j, 1, nbr_idx.unsqueeze(-1).expand(-1, -1, -1, z_emb.size(-1)))
        bias_nbr = self._pair_bias(rbf, z_i, z_j)

        bias_atom_cell = None
        bias_cell_atom = None
        if self.bias_atom_cell is not None:
            cond_exp = cond.unsqueeze(1).expand(-1, atom_tokens.size(1), -1)
            atom_cell = torch.cat([atom_tokens, cell_token.expand(-1, atom_tokens.size(1), -1), cond_exp], dim=-1)
            bias_atom_cell = self.bias_atom_cell(atom_cell).permute(0, 2, 1).unsqueeze(-1)
            cell_atom = torch.cat([cell_token.expand(-1, atom_tokens.size(1), -1), atom_tokens, cond_exp], dim=-1)
            bias_cell_atom = self.bias_cell_atom(cell_atom).permute(0, 2, 1).unsqueeze(2)

        for blk in self.blocks:
            tokens = blk(tokens, cond, nbr_idx, nbr_mask, atom_mask, bias_nbr, bias_atom_cell, bias_cell_atom)
            tokens = tokens * torch.cat([torch.ones(bsz, 1, device=atom_mask.device), atom_mask], dim=1).unsqueeze(-1)

        tokens = self.norm(tokens)
        cell_out = tokens[:, 0, :]
        atom_out = tokens[:, 1:, :]
        pred_f = self.head_f(atom_out)
        pred_z = self.head_z(atom_out)
        pred_g = self.head_g(cell_out)
        return pred_f, pred_g, pred_z


__all__ = ["AtomTransformerConfig", "AtomTransformer"]
