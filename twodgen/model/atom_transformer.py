from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from twodgen.common.crystal import (
    build_knn,
    cholesky6_to_lattice,
    frac_mic_dist,
    frac_mic_dist_with_shifts,
    gram6_to_lattice,
    rbf_expand,
)
from twodgen.data.torus import torus_encode


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
    dual_graph: bool = False
    edge_type_dim: int = 0
    edge_type_gating: bool = True
    wrap_embed_dim: int = 0
    cell_rep: str = "gram6"  # gram6 | cholesky6
    chol_log_min: Optional[float] = None
    chol_log_max: Optional[float] = None
    chol_log_min_vec: Optional[Tuple[float, float, float]] = None
    chol_log_max_vec: Optional[Tuple[float, float, float]] = None
    cond_dim: int = 0
    use_comp_encoder: bool = True
    comp_embed_dim: int = 64
    comp_pool_mode: str = "count"  # count | sqrt | frac
    comp_use_frac: bool = True
    element_ids: Optional[List[int]] = None
    pbc_mask: Tuple[int, int, int] = (1, 1, 0)
    tail_adapter: str = "none"  # none | egnn
    tail_hidden_dim: int = 128
    tail_scale: float = 0.1


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
        self.geom_atom_mlp = nn.Sequential(
            nn.Linear(5, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.geom_cell_mlp = nn.Sequential(
            nn.Linear(3, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(1, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.cond_mlp = None
        if cfg.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cfg.cond_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
        self.comp_embed = None
        self.comp_mlp = None
        if cfg.use_comp_encoder:
            if cfg.element_ids is None:
                element_ids = torch.arange(1, cfg.num_elements + 1, dtype=torch.long)
            else:
                element_ids = torch.tensor(cfg.element_ids, dtype=torch.long)
                if element_ids.numel() != cfg.num_elements:
                    raise ValueError("element_ids length must match num_elements.")
            self.register_buffer("element_ids", element_ids)
            self.comp_embed = nn.Embedding(cfg.num_elements + 1, cfg.comp_embed_dim, padding_idx=0)
            comp_in = cfg.comp_embed_dim + 1
            if cfg.comp_use_frac:
                comp_in += cfg.comp_embed_dim
            self.comp_mlp = nn.Sequential(
                nn.Linear(comp_in, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
        self.cond_scale_time = nn.Parameter(torch.tensor(1.0))
        self.cond_scale_vec = nn.Parameter(torch.tensor(1.0))
        self.cond_scale_comp = nn.Parameter(torch.tensor(1.0))
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
        self.head_uv = nn.Linear(cfg.embed_dim, 4)
        self.head_zn = nn.Linear(cfg.embed_dim, 1)
        self.head_lat = nn.Linear(cfg.embed_dim, 3)
        self.head_t = nn.Linear(cfg.embed_dim, 1)

        pair_in = cfg.rbf_dim + 2 * cfg.z_embed_dim
        if cfg.edge_type_dim > 0:
            pair_in += cfg.edge_type_dim
        if cfg.wrap_embed_dim > 0:
            pair_in += cfg.wrap_embed_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, cfg.pair_mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.pair_mlp_hidden, cfg.num_heads),
        )
        self.edge_type_embed = None
        if cfg.edge_type_dim > 0:
            self.edge_type_embed = nn.Embedding(3, cfg.edge_type_dim)
        self.edge_type_scale = nn.Parameter(torch.ones(3)) if cfg.edge_type_gating else None
        self.wrap_embed = None
        if cfg.wrap_embed_dim > 0:
            self.wrap_embed = nn.Embedding(9, cfg.wrap_embed_dim)
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

    def _pair_bias(
        self,
        rbf: torch.Tensor,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        wrap_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, n, k, _ = rbf.shape
        feats = [rbf, z_i, z_j]
        if self.edge_type_embed is not None:
            if edge_type is None:
                edge_type = torch.zeros((bsz, n, k), device=rbf.device, dtype=torch.long)
            feats.append(self.edge_type_embed(edge_type))
        if self.wrap_embed is not None:
            if wrap_id is None:
                wrap_id = torch.full((bsz, n, k), 4, device=rbf.device, dtype=torch.long)
            feats.append(self.wrap_embed(wrap_id))
        feat = torch.cat(feats, dim=-1)
        bias = self.pair_mlp(feat)
        bias = bias.permute(0, 3, 1, 2)
        if self.edge_type_scale is not None:
            if edge_type is None:
                edge_type = torch.zeros((bsz, n, k), device=rbf.device, dtype=torch.long)
            scale = self.edge_type_scale[edge_type].unsqueeze(1)
            bias = bias * scale
        return bias

    @staticmethod
    def _merge_dual_graph(
        idx_xy: torch.Tensor,
        mask_xy: torch.Tensor,
        dist_xy: torch.Tensor,
        wrap_xy: torch.Tensor,
        idx_3d: torch.Tensor,
        mask_3d: torch.Tensor,
        dist_3d: torch.Tensor,
        wrap_3d: torch.Tensor,
        max_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n, _ = idx_xy.shape
        device = idx_xy.device
        idx_out = torch.zeros((bsz, n, max_k), device=device, dtype=idx_xy.dtype)
        mask_out = torch.zeros((bsz, n, max_k), device=device, dtype=torch.bool)
        edge_type = torch.zeros((bsz, n, max_k), device=device, dtype=torch.long)
        dist_out = torch.zeros((bsz, n, max_k), device=device, dtype=dist_xy.dtype)
        wrap_id = torch.full((bsz, n, max_k), 4, device=device, dtype=torch.long)

        for b in range(bsz):
            for i in range(n):
                neighbors: list[int] = []
                types: list[int] = []
                dists: list[float] = []
                wraps: list[int] = []
                for j in range(idx_xy.shape[-1]):
                    if not mask_xy[b, i, j]:
                        continue
                    nbr = int(idx_xy[b, i, j].item())
                    neighbors.append(nbr)
                    types.append(0)
                    dists.append(float(dist_xy[b, i, j].item()))
                    wraps.append(int(wrap_xy[b, i, j].item()))
                for j in range(idx_3d.shape[-1]):
                    if not mask_3d[b, i, j]:
                        continue
                    nbr = int(idx_3d[b, i, j].item())
                    if nbr in neighbors:
                        pos = neighbors.index(nbr)
                        types[pos] = 2
                        dists[pos] = min(dists[pos], float(dist_3d[b, i, j].item()))
                        wraps[pos] = int(wrap_3d[b, i, j].item())
                    else:
                        neighbors.append(nbr)
                        types.append(1)
                        dists.append(float(dist_3d[b, i, j].item()))
                        wraps.append(int(wrap_3d[b, i, j].item()))
                keep = min(len(neighbors), max_k)
                if keep == 0:
                    continue
                idx_out[b, i, :keep] = torch.tensor(neighbors[:keep], device=device, dtype=idx_xy.dtype)
                edge_type[b, i, :keep] = torch.tensor(types[:keep], device=device, dtype=edge_type.dtype)
                dist_out[b, i, :keep] = torch.tensor(dists[:keep], device=device, dtype=dist_xy.dtype)
                wrap_id[b, i, :keep] = torch.tensor(wraps[:keep], device=device, dtype=wrap_id.dtype)
                mask_out[b, i, :keep] = True
        return idx_out, mask_out, dist_out, edge_type, wrap_id

    @staticmethod
    def _wrap_id_from_shift(shift: torch.Tensor) -> torch.Tensor:
        m = shift[..., 0].clamp(-1, 1).to(torch.long)
        n = shift[..., 1].clamp(-1, 1).to(torch.long)
        return (m + 1) * 3 + (n + 1)

    def _encode_composition(self, counts_vector: torch.Tensor) -> torch.Tensor:
        if self.comp_embed is None or self.comp_mlp is None:
            raise ValueError("Composition encoder requested but not initialized.")
        counts = counts_vector.float()
        if counts.size(-1) != self.element_ids.numel():
            raise ValueError("counts_vector size does not match element_ids length.")
        total = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        elem_ids = self.element_ids.to(counts.device)
        emb = self.comp_embed(elem_ids)
        if self.cfg.comp_pool_mode == "sqrt":
            weights = counts.sqrt()
        elif self.cfg.comp_pool_mode == "frac":
            weights = counts / total
        else:
            weights = counts
        comp_vec = weights @ emb
        parts = [comp_vec]
        if self.cfg.comp_use_frac:
            frac_vec = (counts / total) @ emb
            parts.append(frac_vec)
        parts.append(total)
        return self.comp_mlp(torch.cat(parts, dim=-1))

    def forward(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        g: torch.Tensor,
        atom_mask: torch.Tensor,
        timesteps: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
        uv_angle: Optional[torch.Tensor] = None,
        z_norm: Optional[torch.Tensor] = None,
        lattice_param: Optional[torch.Tensor] = None,
        slab_t: Optional[torch.Tensor] = None,
        return_geom: bool = False,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        dist_nbr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        bsz, n, _ = frac.shape
        t_emb = sinusoidal_timestep_embedding(timesteps, self.cfg.time_embed_dim)
        cond_time = self.time_mlp(t_emb)
        cond = self.cond_scale_time * cond_time
        if self.cond_mlp is not None:
            if cond_vec is None:
                cond_vec = torch.zeros(bsz, self.cfg.cond_dim, device=cond.device, dtype=cond.dtype)
            cond_vec_proj = self.cond_mlp(cond_vec)
            cond = cond + self.cond_scale_vec * cond_vec_proj
        if self.cfg.use_comp_encoder and counts_vector is not None:
            cond_comp = self._encode_composition(counts_vector)
            cond = cond + self.cond_scale_comp * cond_comp

        z_emb = self.z_embed(z)
        f_emb = self.f_proj(torus_encode(frac, self.cfg.fourier_freqs))
        atom_tokens = self.in_proj(torch.cat([z_emb, f_emb], dim=-1))
        if uv_angle is not None or z_norm is not None:
            if uv_angle is None:
                uv_angle = torch.zeros(bsz, n, 4, device=atom_tokens.device, dtype=atom_tokens.dtype)
            if z_norm is None:
                z_norm = torch.zeros(bsz, n, device=atom_tokens.device, dtype=atom_tokens.dtype)
            geom_atom = torch.cat([uv_angle, z_norm.unsqueeze(-1)], dim=-1)
            atom_tokens = atom_tokens + self.geom_atom_mlp(geom_atom)

        cell_token = self.cell_mlp(g).unsqueeze(1)
        if lattice_param is not None:
            cell_token = cell_token + self.geom_cell_mlp(lattice_param).unsqueeze(1)
        if slab_t is not None:
            if slab_t.ndim == 1:
                slab_t = slab_t.unsqueeze(-1)
            cell_token = cell_token + self.t_mlp(slab_t).unsqueeze(1)
        tokens = torch.cat([cell_token, atom_tokens], dim=1)

        use_cache = self.cfg.cache_neighbors and not self.training
        if nbr_idx is not None:
            use_cache = False
        cache_keys = ("nbr_idx", "nbr_mask", "dist_nbr")
        if self.cfg.dual_graph:
            cache_keys = cache_keys + ("edge_type",)
        if self.wrap_embed is not None:
            cache_keys = cache_keys + ("wrap_id",)
        if use_cache and cache_every is not None:
            if step is None or step % max(cache_every, 1) == 0:
                use_cache = False
            else:
                use_cache = all(key in self._cache for key in cache_keys)
        if use_cache and cache_every is None:
            if "frac" in self._cache and "g" in self._cache:
                prev_frac = self._cache["frac"]
                prev_g = self._cache["g"]
                if torch.mean((prev_frac - frac) ** 2) < self.cfg.cache_thresh and torch.mean((prev_g - g) ** 2) < self.cfg.cache_thresh:
                    use_cache = all(key in self._cache for key in cache_keys)
                else:
                    use_cache = False
            else:
                use_cache = False
        edge_type = None
        wrap_id = None
        need_wrap = self.wrap_embed is not None
        if nbr_idx is not None:
            if nbr_mask is None:
                raise ValueError("nbr_idx requires nbr_mask.")
            dist = None
        elif not use_cache:
            if self.cfg.cell_rep == "cholesky6":
                log_min = self.cfg.chol_log_min_vec if self.cfg.chol_log_min_vec is not None else self.cfg.chol_log_min
                log_max = self.cfg.chol_log_max_vec if self.cfg.chol_log_max_vec is not None else self.cfg.chol_log_max
                lattice = cholesky6_to_lattice(g, log_min=log_min, log_max=log_max)
                lattice = lattice * self.cfg.g_scale ** 0.5
            else:
                lattice = gram6_to_lattice(g * self.cfg.g_scale)
            if self.cfg.dual_graph:
                lattice_xy = lattice.clone()
                lattice_xy[:, 2, :] = 0.0
                pbc_xy = (self.cfg.pbc_mask[0], self.cfg.pbc_mask[1], 0)
                if need_wrap:
                    dist_xy, shift_xy = frac_mic_dist_with_shifts(frac, lattice_xy, atom_mask, pbc_mask=pbc_xy)
                    dist_3d, shift_3d = frac_mic_dist_with_shifts(frac, lattice, atom_mask, pbc_mask=self.cfg.pbc_mask)
                else:
                    dist_xy = frac_mic_dist(frac, lattice_xy, atom_mask, pbc_mask=pbc_xy)
                    dist_3d = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=self.cfg.pbc_mask)
                idx_xy, mask_xy = build_knn(dist_xy, self.cfg.k_neighbors)
                dist_xy_nbr = torch.gather(dist_xy, 2, idx_xy)
                idx_3d, mask_3d = build_knn(dist_3d, self.cfg.k_neighbors)
                dist_3d_nbr = torch.gather(dist_3d, 2, idx_3d)
                wrap_xy = None
                wrap_3d = None
                if need_wrap:
                    shift_xy_nbr = torch.gather(
                        shift_xy, 2, idx_xy.unsqueeze(-1).expand(-1, -1, -1, 3)
                    )
                    shift_3d_nbr = torch.gather(
                        shift_3d, 2, idx_3d.unsqueeze(-1).expand(-1, -1, -1, 3)
                    )
                    wrap_xy = self._wrap_id_from_shift(shift_xy_nbr)
                    wrap_3d = self._wrap_id_from_shift(shift_3d_nbr)
                max_k = min(frac.shape[1], idx_xy.shape[-1] + idx_3d.shape[-1])
                nbr_idx, nbr_mask, dist_nbr, edge_type, wrap_id = self._merge_dual_graph(
                    idx_xy,
                    mask_xy,
                    dist_xy_nbr,
                    wrap_xy if wrap_xy is not None else torch.full_like(dist_xy_nbr, 4, dtype=torch.long),
                    idx_3d,
                    mask_3d,
                    dist_3d_nbr,
                    wrap_3d if wrap_3d is not None else torch.full_like(dist_3d_nbr, 4, dtype=torch.long),
                    max_k,
                )
            else:
                if need_wrap:
                    dist, shift = frac_mic_dist_with_shifts(frac, lattice, atom_mask, pbc_mask=self.cfg.pbc_mask)
                else:
                    dist = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=self.cfg.pbc_mask)
                nbr_idx, nbr_mask = build_knn(dist, self.cfg.k_neighbors)
                dist_nbr = torch.gather(dist, 2, nbr_idx)
                if need_wrap:
                    shift_nbr = torch.gather(shift, 2, nbr_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
                    wrap_id = self._wrap_id_from_shift(shift_nbr)
            if self.cfg.cache_neighbors:
                self._cache["frac"] = frac.detach()
                self._cache["g"] = g.detach()
                self._cache["nbr_idx"] = nbr_idx.detach()
                self._cache["nbr_mask"] = nbr_mask.detach()
                self._cache["dist_nbr"] = dist_nbr.detach()
                if edge_type is not None:
                    self._cache["edge_type"] = edge_type.detach()
                if wrap_id is not None:
                    self._cache["wrap_id"] = wrap_id.detach()
        else:
            nbr_idx = self._cache["nbr_idx"]
            nbr_mask = self._cache["nbr_mask"]
            dist_nbr = self._cache["dist_nbr"]
            edge_type = self._cache.get("edge_type")
            wrap_id = self._cache.get("wrap_id")
        if dist_nbr is None:
            if self.cfg.cell_rep == "cholesky6":
                log_min = self.cfg.chol_log_min_vec if self.cfg.chol_log_min_vec is not None else self.cfg.chol_log_min
                log_max = self.cfg.chol_log_max_vec if self.cfg.chol_log_max_vec is not None else self.cfg.chol_log_max
                lattice = cholesky6_to_lattice(g, log_min=log_min, log_max=log_max)
                lattice = lattice * self.cfg.g_scale ** 0.5
            else:
                lattice = gram6_to_lattice(g * self.cfg.g_scale)
            dist = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=self.cfg.pbc_mask)
            dist_nbr = torch.gather(dist, 2, nbr_idx)
        if nbr_mask.dtype != torch.bool:
            nbr_mask = nbr_mask > 0.5
        rbf = rbf_expand(dist_nbr, self.cfg.rbf_dim, self.cfg.rbf_r_max)
        z_i = z_emb.unsqueeze(2).expand(-1, -1, nbr_idx.shape[-1], -1)
        z_j = z_emb.unsqueeze(2).expand(-1, -1, nbr_idx.shape[-1], -1)
        z_j = torch.gather(z_j, 1, nbr_idx.unsqueeze(-1).expand(-1, -1, -1, z_emb.size(-1)))
        bias_nbr = self._pair_bias(rbf, z_i, z_j, edge_type=edge_type, wrap_id=wrap_id)

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
        if return_geom:
            pred_uv = self.head_uv(atom_out)
            pred_zn = self.head_zn(atom_out).squeeze(-1)
            pred_lat = self.head_lat(cell_out)
            pred_t = self.head_t(cell_out).squeeze(-1)
            return pred_f, pred_g, pred_z, {
                "uv_angle": pred_uv,
                "z_norm": pred_zn,
                "lattice_param": pred_lat,
                "t": pred_t,
            }
        return pred_f, pred_g, pred_z


__all__ = ["AtomTransformerConfig", "AtomTransformer"]
