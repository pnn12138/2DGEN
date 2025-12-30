from __future__ import annotations

from typing import Optional, Tuple

import torch




def gram6_to_lattice(
    g: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 5,
    fallback_eig: bool = True,
) -> torch.Tensor:
    """
    Convert Gram 6D vectors to lattice matrices using Cholesky with jitter fallback.
    """
    if g.ndim != 2 or g.shape[-1] != 6:
        raise ValueError(f"Expected g shape (B,6), got {tuple(g.shape)}")
    bsz = g.shape[0]
    device = g.device
    dtype = g.dtype

    lattices = []
    eye = torch.eye(3, device=device, dtype=dtype)
    for i in range(bsz):
        g_i = g[i]
        G = torch.stack(
            [
                torch.stack([g_i[0], g_i[3], g_i[4]], dim=0),
                torch.stack([g_i[3], g_i[1], g_i[5]], dim=0),
                torch.stack([g_i[4], g_i[5], g_i[2]], dim=0),
            ],
            dim=0,
        )
        G = 0.5 * (G + G.transpose(0, 1))
        L = None
        eps = jitter
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(G + eps * eye)
                break
            except Exception:
                eps *= 10.0
        if L is None and fallback_eig:
            vals, vecs = torch.linalg.eigh(G)
            vals = torch.clamp(vals, min=jitter)
            G = vecs.matmul(torch.diag(vals)).matmul(vecs.transpose(0, 1))
            try:
                L = torch.linalg.cholesky(G + jitter * eye)
            except Exception:
                L = None
        if L is None:
            L = torch.linalg.cholesky(eye * jitter)
        lattices.append(L)
    return torch.stack(lattices, dim=0)


def gram6_to_cholesky6(
    g: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 5,
    fallback_eig: bool = True,
    log_min: Optional[float] = None,
    log_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert Gram 6D vectors to Cholesky-6D parameters.
    y = [log(r11), log(r22), log(r33), r12, r13, r23] from g = R^T R.
    """
    if g.ndim != 2 or g.shape[-1] != 6:
        raise ValueError(f"Expected g shape (B,6), got {tuple(g.shape)}")
    bsz = g.shape[0]
    device = g.device
    dtype = g.dtype

    ys = []
    eye = torch.eye(3, device=device, dtype=dtype)
    for i in range(bsz):
        g_i = g[i]
        G = torch.stack(
            [
                torch.stack([g_i[0], g_i[3], g_i[4]], dim=0),
                torch.stack([g_i[3], g_i[1], g_i[5]], dim=0),
                torch.stack([g_i[4], g_i[5], g_i[2]], dim=0),
            ],
            dim=0,
        )
        G = 0.5 * (G + G.transpose(0, 1))
        L = None
        eps = jitter
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(G + eps * eye)
                break
            except Exception:
                eps *= 10.0
        if L is None and fallback_eig:
            vals, vecs = torch.linalg.eigh(G)
            vals = torch.clamp(vals, min=jitter)
            G = vecs.matmul(torch.diag(vals)).matmul(vecs.transpose(0, 1))
            try:
                L = torch.linalg.cholesky(G + jitter * eye)
            except Exception:
                L = None
        if L is None:
            L = torch.linalg.cholesky(eye * jitter)

        diag = torch.log(torch.diag(L))
        if log_min is not None or log_max is not None:
            diag = torch.clamp(diag, min=log_min, max=log_max)
        y = torch.stack([diag[0], diag[1], diag[2], L[1, 0], L[2, 0], L[2, 1]], dim=0)
        ys.append(y)
    return torch.stack(ys, dim=0)


def cholesky6_to_gram6(
    y: torch.Tensor,
    log_min: Optional[float] = None,
    log_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Decode Cholesky-6D parameters to Gram 6D vector.
    """
    if y.ndim != 2 or y.shape[-1] != 6:
        raise ValueError(f"Expected y shape (B,6), got {tuple(y.shape)}")
    diag = y[:, :3]
    if log_min is not None or log_max is not None:
        diag = torch.clamp(diag, min=log_min, max=log_max)
    r11, r22, r33 = torch.exp(diag[:, 0]), torch.exp(diag[:, 1]), torch.exp(diag[:, 2])
    r12, r13, r23 = y[:, 3], y[:, 4], y[:, 5]

    R = torch.zeros((y.shape[0], 3, 3), device=y.device, dtype=y.dtype)
    R[:, 0, 0] = r11
    R[:, 0, 1] = r12
    R[:, 0, 2] = r13
    R[:, 1, 1] = r22
    R[:, 1, 2] = r23
    R[:, 2, 2] = r33

    G = R.transpose(-1, -2).matmul(R)
    g = torch.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1], G[:, 0, 2], G[:, 1, 2]], dim=-1)
    return g


def cholesky6_to_lattice(
    y: torch.Tensor,
    log_min: Optional[float] = None,
    log_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Decode Cholesky-6D parameters to lattice matrices (lower-triangular).
    """
    if y.ndim != 2 or y.shape[-1] != 6:
        raise ValueError(f"Expected y shape (B,6), got {tuple(y.shape)}")
    diag = y[:, :3]
    if log_min is not None or log_max is not None:
        diag = torch.clamp(diag, min=log_min, max=log_max)
    r11, r22, r33 = torch.exp(diag[:, 0]), torch.exp(diag[:, 1]), torch.exp(diag[:, 2])
    r12, r13, r23 = y[:, 3], y[:, 4], y[:, 5]

    R = torch.zeros((y.shape[0], 3, 3), device=y.device, dtype=y.dtype)
    R[:, 0, 0] = r11
    R[:, 0, 1] = r12
    R[:, 0, 2] = r13
    R[:, 1, 1] = r22
    R[:, 1, 2] = r23
    R[:, 2, 2] = r33
    return R.transpose(-1, -2)


def reduce_lattice_simple(lattice: torch.Tensor) -> torch.Tensor:
    """
    Simple reduction: sort basis vectors by length and enforce right-handedness.
    """
    lengths = torch.linalg.norm(lattice, dim=-1)
    order = torch.argsort(lengths, dim=-1)
    sorted_lattice = torch.gather(
        lattice,
        -2,
        order.unsqueeze(-1).expand(-1, -1, 3),
    )
    det = torch.linalg.det(sorted_lattice)
    flip = det < 0
    if flip.any():
        sorted_lattice[flip, 0, :] = -sorted_lattice[flip, 0, :]
    return sorted_lattice


def clip_lattice(
    lattice: torch.Tensor,
    v_min: float,
    v_max: float,
    cond_max: float,
) -> torch.Tensor:
    """
    Clip lattice volume and condition number by isotropic scaling.
    """
    volume = torch.abs(torch.linalg.det(lattice))
    v_target = volume.clone()
    v_target = torch.clamp(v_target, min=v_min, max=v_max)
    scale = (v_target / volume.clamp_min(1e-8)).pow(1.0 / 3.0)
    lattice = lattice * scale.unsqueeze(-1).unsqueeze(-1)

    gram = lattice.transpose(-1, -2).matmul(lattice)
    eigvals = torch.linalg.eigvalsh(gram)
    cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-8)
    mask = cond > cond_max
    if mask.any():
        scale_down = (cond_max / cond[mask]).pow(0.5)
        lattice[mask] = lattice[mask] * scale_down.unsqueeze(-1).unsqueeze(-1)
    return lattice


def frac_mic_dist(
    frac: torch.Tensor,
    lattice: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute pairwise MIC distances for fractional coords.

    Args:
        frac: (B, N, 3) fractional coords.
        lattice: (B, 3, 3) lattice matrices.
        mask: (B, N) atom mask (1 real, 0 pad).
    Returns:
        dist: (B, N, N) with PAD/self filled as +inf.
    """
    df = frac[:, :, None, :] - frac[:, None, :, :]
    df_mic = df - torch.round(df)
    dr = torch.einsum("bijn,bnm->bijm", df_mic, lattice)
    dist = torch.linalg.norm(dr, dim=-1)

    valid = mask > 0.5
    inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
    dist = dist.masked_fill(~valid[:, :, None], inf)
    dist = dist.masked_fill(~valid[:, None, :], inf)

    n = dist.shape[-1]
    idx = torch.arange(n, device=dist.device)
    dist[:, idx, idx] = inf
    return dist.clamp_min(eps)


def build_knn(
    dist: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN indices and mask from distance matrix.

    Returns:
        idx: (B, N, k) indices into atom dimension.
        mask: (B, N, k) valid neighbor mask.
    """
    bsz, n, _ = dist.shape
    k = min(k, n)
    vals, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    mask = torch.isfinite(vals)
    return idx, mask


def rbf_expand(dist: torch.Tensor, num_rbf: int, r_max: float) -> torch.Tensor:
    """
    Gaussian RBF expansion for distances.
    """
    if num_rbf <= 1:
        raise ValueError("num_rbf must be > 1")
    centers = torch.linspace(0.0, r_max, num_rbf, device=dist.device, dtype=dist.dtype)
    width = (r_max / (num_rbf - 1)) ** 2
    diff = dist.unsqueeze(-1) - centers
    return torch.exp(-diff * diff / (width + 1e-8))


def niggli_reduce_lattice(lattice: torch.Tensor) -> torch.Tensor:
    """
    Niggli-reduce lattice matrices using pymatgen.
    """
    try:
        from pymatgen.core import Lattice
    except ImportError as exc:
        raise ImportError("pymatgen is required for Niggli reduction") from exc

    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattice shape (B,3,3), got {tuple(lattice.shape)}")
    device = lattice.device
    dtype = lattice.dtype
    latt_np = lattice.detach().cpu().numpy()
    reduced = [Lattice(mat).get_niggli_reduced_lattice().matrix for mat in latt_np]
    return torch.tensor(reduced, device=device, dtype=dtype)


__all__ = [
    "gram6_to_lattice",
    "gram6_to_cholesky6",
    "cholesky6_to_gram6",
    "cholesky6_to_lattice",
    "reduce_lattice_simple",
    "niggli_reduce_lattice",
    "clip_lattice",
    "frac_mic_dist",
    "build_knn",
    "rbf_expand",
]
