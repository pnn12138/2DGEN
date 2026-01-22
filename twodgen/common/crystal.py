from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch


_DiagBound = Union[float, torch.Tensor, Sequence[float]]


def _as_diag_bound(
    value: Optional[_DiagBound],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Union[float, torch.Tensor]]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        if value.shape == (3,):
            return value.to(device=device, dtype=dtype)
        raise ValueError(f"Expected log bound tensor shape (3,) or scalar, got {tuple(value.shape)}")
    vals = list(value)
    if len(vals) != 3:
        raise ValueError(f"Expected log bound sequence length 3, got {len(vals)}")
    return torch.tensor([float(v) for v in vals], device=device, dtype=dtype)


def _clamp_diag(
    diag: torch.Tensor,
    *,
    log_min: Optional[_DiagBound],
    log_max: Optional[_DiagBound],
) -> torch.Tensor:
    if log_min is None and log_max is None:
        return diag
    min_bound = _as_diag_bound(log_min, device=diag.device, dtype=diag.dtype)
    max_bound = _as_diag_bound(log_max, device=diag.device, dtype=diag.dtype)
    if min_bound is not None and max_bound is not None:
        return torch.clamp(diag, min=min_bound, max=max_bound)
    if min_bound is not None:
        return torch.clamp(diag, min=min_bound)
    return torch.clamp(diag, max=max_bound)




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
    log_min: Optional[_DiagBound] = None,
    log_max: Optional[_DiagBound] = None,
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
        diag = _clamp_diag(diag, log_min=log_min, log_max=log_max)
        y = torch.stack([diag[0], diag[1], diag[2], L[1, 0], L[2, 0], L[2, 1]], dim=0)
        ys.append(y)
    return torch.stack(ys, dim=0)


def cholesky6_to_gram6(
    y: torch.Tensor,
    log_min: Optional[_DiagBound] = None,
    log_max: Optional[_DiagBound] = None,
) -> torch.Tensor:
    """
    Decode Cholesky-6D parameters to Gram 6D vector.
    """
    if y.ndim != 2 or y.shape[-1] != 6:
        raise ValueError(f"Expected y shape (B,6), got {tuple(y.shape)}")
    diag = y[:, :3]
    diag = _clamp_diag(diag, log_min=log_min, log_max=log_max)
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
    log_min: Optional[_DiagBound] = None,
    log_max: Optional[_DiagBound] = None,
) -> torch.Tensor:
    """
    Decode Cholesky-6D parameters to lattice matrices (lower-triangular).
    """
    if y.ndim != 2 or y.shape[-1] != 6:
        raise ValueError(f"Expected y shape (B,6), got {tuple(y.shape)}")
    diag = y[:, :3]
    diag = _clamp_diag(diag, log_min=log_min, log_max=log_max)
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


def lattice_to_gram6(lattice: torch.Tensor) -> torch.Tensor:
    """
    Convert lattice matrices to Gram 6D vectors.
    """
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattice shape (B,3,3), got {tuple(lattice.shape)}")
    # Convention: lattice basis vectors live in rows, and Cartesian coordinates are
    # computed as `cart = frac @ lattice`. Under this convention the Gram matrix is
    # `G = lattice @ lattice^T`.
    gram = lattice.matmul(lattice.transpose(-1, -2))
    return torch.stack(
        [gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2], gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]],
        dim=-1,
    )


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

    gram = lattice.matmul(lattice.transpose(-1, -2))
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
    pbc_mask: Optional[Tuple[int, int, int]] = None,
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
    df = frac[:, :, None, :] - frac[:, None, :, :]  # (B, N, N, 3)

    # Exact MIC via enumerating neighbor-cell shifts.
    # For 2D PBC: 9 shifts; for 3D PBC: 27 shifts. With max_atoms<=24 this is cheap and avoids
    # incorrect rounding behavior under non-orthogonal (skewed) cells.
    if pbc_mask is None:
        pbc = torch.ones((3,), device=df.device, dtype=torch.long)
    else:
        pbc = torch.tensor(pbc_mask, device=df.device, dtype=torch.long)
        if pbc.shape != (3,):
            raise ValueError("pbc_mask must be a length-3 tuple.")
        if not torch.all((pbc == 0) | (pbc == 1)):
            raise ValueError("pbc_mask values must be 0 or 1.")

    shifts_1d = torch.tensor([-1, 0, 1], device=df.device, dtype=df.dtype)
    zeros_1d = torch.tensor([0], device=df.device, dtype=df.dtype)
    # Only enumerate periodic dimensions to avoid redundant work (2D slab: 9 shifts).
    components = [
        shifts_1d if int(pbc[0].item()) == 1 else zeros_1d,
        shifts_1d if int(pbc[1].item()) == 1 else zeros_1d,
        shifts_1d if int(pbc[2].item()) == 1 else zeros_1d,
    ]
    shifts = torch.cartesian_prod(*components)  # (S, 3)

    df_shifted = df.unsqueeze(-2) - shifts.view(1, 1, 1, -1, 3)  # (B, N, N, S, 3)
    dr = torch.einsum("bijsk,bkm->bijsm", df_shifted, lattice)  # (B, N, N, S, 3)
    dist_all = torch.linalg.norm(dr, dim=-1)  # (B, N, N, S)
    dist = dist_all.min(dim=-1).values  # (B, N, N)

    valid = mask > 0.5
    inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
    dist = dist.masked_fill(~valid[:, :, None], inf)
    dist = dist.masked_fill(~valid[:, None, :], inf)

    # Avoid advanced-indexing pitfalls on older/newer PyTorch versions.
    dist.diagonal(dim1=-2, dim2=-1).fill_(inf)
    return dist.clamp_min(eps)


def frac_mic_dist_with_shifts(
    frac: torch.Tensor,
    lattice: torch.Tensor,
    mask: torch.Tensor,
    pbc_mask: Optional[Tuple[int, int, int]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MIC distances and the corresponding shift vectors (m,n,l).

    Returns:
        dist: (B, N, N) with PAD/self filled as +inf.
        shifts: (B, N, N, 3) integer shifts in {-1,0,1} (or 0 on non-PBC axes).
    """
    df = frac[:, :, None, :] - frac[:, None, :, :]  # (B, N, N, 3)

    if pbc_mask is None:
        pbc = torch.ones((3,), device=df.device, dtype=torch.long)
    else:
        pbc = torch.tensor(pbc_mask, device=df.device, dtype=torch.long)
        if pbc.shape != (3,):
            raise ValueError("pbc_mask must be a length-3 tuple.")
        if not torch.all((pbc == 0) | (pbc == 1)):
            raise ValueError("pbc_mask values must be 0 or 1.")

    shifts_1d = torch.tensor([-1, 0, 1], device=df.device, dtype=df.dtype)
    zeros_1d = torch.tensor([0], device=df.device, dtype=df.dtype)
    components = [
        shifts_1d if int(pbc[0].item()) == 1 else zeros_1d,
        shifts_1d if int(pbc[1].item()) == 1 else zeros_1d,
        shifts_1d if int(pbc[2].item()) == 1 else zeros_1d,
    ]
    shifts = torch.cartesian_prod(*components)  # (S, 3)

    df_shifted = df.unsqueeze(-2) - shifts.view(1, 1, 1, -1, 3)  # (B, N, N, S, 3)
    dr = torch.einsum("bijsk,bkm->bijsm", df_shifted, lattice)  # (B, N, N, S, 3)
    dist_all = torch.linalg.norm(dr, dim=-1)  # (B, N, N, S)
    min_idx = dist_all.argmin(dim=-1)  # (B, N, N)
    dist = dist_all.gather(-1, min_idx.unsqueeze(-1)).squeeze(-1)

    shifts_idx = shifts.to(torch.long)
    shifts_selected = shifts_idx[min_idx]

    valid = mask > 0.5
    inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
    dist = dist.masked_fill(~valid[:, :, None], inf)
    dist = dist.masked_fill(~valid[:, None, :], inf)
    dist.diagonal(dim1=-2, dim2=-1).fill_(inf)
    shifts_selected = shifts_selected * valid[:, :, None].unsqueeze(-1)
    shifts_selected = shifts_selected * valid[:, None, :].unsqueeze(-1)
    return dist.clamp_min(eps), shifts_selected


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
    "lattice_to_gram6",
    "reduce_lattice_simple",
    "niggli_reduce_lattice",
    "clip_lattice",
    "frac_mic_dist",
    "frac_mic_dist_with_shifts",
    "build_knn",
    "rbf_expand",
]
