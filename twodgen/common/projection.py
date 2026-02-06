from __future__ import annotations

"""
Sampling-time hard projection utilities.

These are intentionally "engineering guardrails" and are meant to be used under
`torch.no_grad()` during sampling, not training. They are not designed to be
fully differentiable.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CellParams:
    a: torch.Tensor  # (B,)
    b: torch.Tensor  # (B,)
    c: torch.Tensor  # (B,)
    alpha: torch.Tensor  # (B,) degrees
    beta: torch.Tensor  # (B,) degrees
    gamma: torch.Tensor  # (B,) degrees


def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.norm(x, dim=-1).clamp_min(eps)


def lattice_to_cell(lattice: torch.Tensor) -> CellParams:
    """
    Args:
        lattice: (B,3,3) row-basis lattice. Cartesian = frac @ lattice.
    """
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattice shape (B,3,3), got {tuple(lattice.shape)}")
    a_vec, b_vec, c_vec = lattice[:, 0], lattice[:, 1], lattice[:, 2]
    a = _safe_norm(a_vec)
    b = _safe_norm(b_vec)
    c = _safe_norm(c_vec)
    eps = 1e-8
    cos_alpha = (b_vec * c_vec).sum(dim=-1) / (b * c).clamp_min(eps)
    cos_beta = (a_vec * c_vec).sum(dim=-1) / (a * c).clamp_min(eps)
    cos_gamma = (a_vec * b_vec).sum(dim=-1) / (a * b).clamp_min(eps)
    cos_alpha = cos_alpha.clamp(-1.0, 1.0)
    cos_beta = cos_beta.clamp(-1.0, 1.0)
    cos_gamma = cos_gamma.clamp(-1.0, 1.0)
    alpha = torch.rad2deg(torch.acos(cos_alpha))
    beta = torch.rad2deg(torch.acos(cos_beta))
    gamma = torch.rad2deg(torch.acos(cos_gamma))
    return CellParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


def _oriented_basis_from_a_b(
    a_vec: torch.Tensor, b_vec: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal basis (e1,e2,e3) from a and b, preserving orientation.
    """
    eps = 1e-8
    e1 = a_vec / _safe_norm(a_vec, eps).unsqueeze(-1)
    b_perp = b_vec - (b_vec * e1).sum(dim=-1, keepdim=True) * e1
    b_perp_norm = _safe_norm(b_perp, eps).unsqueeze(-1)
    # If b nearly collinear with a, pick an arbitrary perpendicular vector.
    collinear = (b_perp_norm.squeeze(-1) <= 10 * eps)
    if torch.any(collinear):
        # Pick a reference axis not parallel to e1.
        ref = torch.tensor([1.0, 0.0, 0.0], device=a_vec.device, dtype=a_vec.dtype).expand_as(e1)
        too_parallel = (torch.abs((ref * e1).sum(dim=-1)) > 0.9)
        ref = torch.where(
            too_parallel.unsqueeze(-1),
            torch.tensor([0.0, 1.0, 0.0], device=a_vec.device, dtype=a_vec.dtype).expand_as(e1),
            ref,
        )
        b_perp_fallback = ref - (ref * e1).sum(dim=-1, keepdim=True) * e1
        b_perp = torch.where(collinear.unsqueeze(-1), b_perp_fallback, b_perp)
        b_perp_norm = _safe_norm(b_perp, eps).unsqueeze(-1)
    e2 = b_perp / b_perp_norm
    e3 = torch.cross(e1, e2, dim=-1)
    e3 = e3 / _safe_norm(e3, eps).unsqueeze(-1)
    return e1, e2, e3


def cell_to_lattice_oriented(
    cell: CellParams,
    *,
    ref_lattice: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reconstruct a lattice using the oriented basis implied by ref_lattice's (a,b).
    This avoids canonical re-orientation that can overly disturb sampled structures.
    """
    if ref_lattice.ndim != 3 or ref_lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ref_lattice shape (B,3,3), got {tuple(ref_lattice.shape)}")
    a0 = ref_lattice[:, 0]
    b0 = ref_lattice[:, 1]
    e1, e2, e3 = _oriented_basis_from_a_b(a0, b0)

    a = cell.a.clamp_min(eps)
    b = cell.b.clamp_min(eps)
    c = cell.c.clamp_min(eps)
    alpha = torch.deg2rad(cell.alpha)
    beta = torch.deg2rad(cell.beta)
    gamma = torch.deg2rad(cell.gamma)

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1e-6)

    a_vec = a.unsqueeze(-1) * e1
    b_vec = b.unsqueeze(-1) * (cos_gamma.unsqueeze(-1) * e1 + sin_gamma.unsqueeze(-1) * e2)

    c_x = cos_beta
    c_y = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_sq = (1.0 - c_x**2 - c_y**2).clamp_min(0.0)
    c_z = torch.sqrt(c_z_sq)
    c_vec = c.unsqueeze(-1) * (c_x.unsqueeze(-1) * e1 + c_y.unsqueeze(-1) * e2 + c_z.unsqueeze(-1) * e3)

    lattice = torch.stack([a_vec, b_vec, c_vec], dim=1)
    det = torch.linalg.det(lattice)
    flip = det < 0
    if flip.any():
        lattice[flip, 0, :] = -lattice[flip, 0, :]
    return lattice


def cond_gram(
    lattice: torch.Tensor,
    *,
    pbc_mask: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """
    Condition number of the Gram matrix (eig_max/eig_min).

    For 2D slabs (pbc_mask has exactly two periodic axes), we compute the in-plane
    Gram cond so the vacuum axis does not dominate cond. For 3D periodic cells
    (or when pbc_mask is None), we fall back to the full 3x3 Gram cond.
    """
    if pbc_mask is not None:
        axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
        if len(axes) == 2:
            sub = lattice[:, axes, :]  # (B,2,3)
            gram = sub @ sub.transpose(-1, -2)  # (B,2,2)
            eigvals = torch.linalg.eigvalsh(gram)
            spd_ok = torch.isfinite(eigvals).all(dim=-1) & (eigvals.min(dim=-1).values > 0.0)
            cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-12)
            return torch.where(
                spd_ok,
                cond,
                torch.tensor(float("inf"), device=lattice.device, dtype=lattice.dtype),
            )

    gram = lattice @ lattice.transpose(-1, -2)
    eigvals = torch.linalg.eigvalsh(gram)
    spd_ok = torch.isfinite(eigvals).all(dim=-1) & (eigvals.min(dim=-1).values > 0.0)
    cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-12)
    return torch.where(spd_ok, cond, torch.tensor(float("inf"), device=lattice.device, dtype=lattice.dtype))


def _angle_out_of_range(cell: CellParams, angle_min: float, angle_max: float) -> torch.Tensor:
    amin = float(angle_min)
    amax = float(angle_max)
    return (
        (cell.alpha < amin)
        | (cell.alpha > amax)
        | (cell.beta < amin)
        | (cell.beta > amax)
        | (cell.gamma < amin)
        | (cell.gamma > amax)
    )


def _inplane_metrics(
    lattice: torch.Tensor,
    *,
    pbc_mask: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (a_len, b_len, gamma_deg) for the in-plane axes defined by pbc_mask==1.
    Assumes exactly two periodic axes.
    """
    axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
    if len(axes) != 2:
        raise ValueError(f"Expected 2 periodic axes for in-plane metrics, got pbc_mask={pbc_mask}")
    a_vec = lattice[:, axes[0], :]
    b_vec = lattice[:, axes[1], :]
    a_len = _safe_norm(a_vec)
    b_len = _safe_norm(b_vec)
    cos_g = (a_vec * b_vec).sum(dim=-1) / (a_len * b_len).clamp_min(1e-8)
    cos_g = cos_g.clamp(-1.0, 1.0)
    gamma = torch.rad2deg(torch.acos(cos_g))
    return a_len, b_len, gamma


def inplane_degenerate(
    lattice: torch.Tensor,
    *,
    pbc_mask: Tuple[int, int, int],
    a_min: float,
    b_min: float,
    gamma_min: float,
    gamma_max: float,
    area_min: float,
) -> torch.Tensor:
    axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
    a_vec = lattice[:, axes[0], :]
    b_vec = lattice[:, axes[1], :]
    a_len, b_len, gamma = _inplane_metrics(lattice, pbc_mask=pbc_mask)
    area = torch.linalg.norm(torch.cross(a_vec, b_vec, dim=-1), dim=-1)
    return (
        (a_len < float(a_min))
        | (b_len < float(b_min))
        | (gamma < float(gamma_min))
        | (gamma > float(gamma_max))
        | (area < float(area_min))
    )


def project_cell_angles_inplace(cell: CellParams, *, angle_min: float, angle_max: float) -> CellParams:
    # Keep a small margin from the boundary to avoid float round-off causing
    # an immediate re-violation in downstream checks.
    eps = 1e-3
    amin = float(angle_min) + eps
    amax = float(angle_max) - eps if float(angle_max) - float(angle_min) > 2 * eps else float(angle_max)
    return CellParams(
        a=cell.a,
        b=cell.b,
        c=cell.c,
        alpha=cell.alpha.clamp(min=amin, max=amax),
        beta=cell.beta.clamp(min=amin, max=amax),
        gamma=cell.gamma.clamp(min=amin, max=amax),
    )


def project_cell_inplane_inplace(
    cell: CellParams,
    *,
    a_min: float,
    b_min: float,
    gamma_min: float,
    gamma_max: float,
    area_min: float,
) -> CellParams:
    a = cell.a.clamp_min(float(a_min))
    b = cell.b.clamp_min(float(b_min))
    eps = 1e-3
    gmin = float(gamma_min) + eps
    gmax = float(gamma_max) - eps if float(gamma_max) - float(gamma_min) > 2 * eps else float(gamma_max)
    gamma = cell.gamma.clamp(min=gmin, max=gmax)
    # Ensure area >= area_min by scaling (a,b) isotropically if needed.
    sin_g = torch.sin(torch.deg2rad(gamma)).clamp_min(1e-6)
    area = a * b * sin_g
    scale = torch.sqrt(torch.tensor(float(area_min), device=area.device, dtype=area.dtype) / area.clamp_min(1e-8))
    scale = torch.where(area < float(area_min), scale, torch.ones_like(scale))
    a = a * scale
    b = b * scale
    return CellParams(a=a, b=b, c=cell.c, alpha=cell.alpha, beta=cell.beta, gamma=gamma)


def project_cond_svd(
    lattice: torch.Tensor,
    *,
    cond_max: float,
    pbc_mask: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """
    Clamp Gram condition number via SVD by enforcing
    sigma_min >= sigma_max / sqrt(cond_max).

    Note:
    - Gram cond = (sigma_max/sigma_min)^2 for the underlying linear map.
    - For 2D slabs (pbc_mask has two periodic axes), we only clamp the in-plane
      submatrix so the vacuum axis does not dominate.
    """
    kappa = float(cond_max)
    if not (kappa > 0.0):
        return lattice
    target = math.sqrt(max(kappa, 1e-12))
    if pbc_mask is not None:
        axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
        if len(axes) == 2:
            sub = lattice[:, axes, :]  # (B,2,3)
            u, s, vh = torch.linalg.svd(sub)  # u:(B,2,2), s:(B,2), vh:(B,3,3)
            s_max = s.max(dim=-1, keepdim=True).values
            s_min_target = s_max / max(float(target), 1e-12)
            s_new = torch.maximum(s, s_min_target)
            sub_proj = u @ torch.diag_embed(s_new) @ vh[:, : s_new.shape[-1], :]
            proj = lattice.clone()
            proj[:, axes, :] = sub_proj
            det = torch.linalg.det(proj)
            flip = det < 0
            if flip.any():
                proj[flip, axes[0], :] = -proj[flip, axes[0], :]
            return proj

    u, s, vt = torch.linalg.svd(lattice)
    s_max = s.max(dim=-1, keepdim=True).values
    s_min_target = s_max / max(float(target), 1e-12)
    s_new = torch.maximum(s, s_min_target)
    proj = u @ torch.diag_embed(s_new) @ vt
    det = torch.linalg.det(proj)
    flip = det < 0
    if flip.any():
        proj[flip, 0, :] = -proj[flip, 0, :]
    return proj


def post_step_project(
    lattice: torch.Tensor,
    *,
    keys: Sequence[str],
    pbc_mask: Tuple[int, int, int],
    angle_min: float,
    angle_max: float,
    cond_max: Optional[float],
    vol_min: Optional[float] = None,
    vol_max: Optional[float] = None,
    inplane_a_min: float,
    inplane_b_min: float,
    inplane_gamma_min: float,
    inplane_gamma_max: float,
    inplane_area_min: float,
    max_iters: int = 2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply a small number of hard projections in a stable order.
    Returns projected lattice and a stats dict with before/after metrics.
    """
    keys_set = {k.strip().lower() for k in keys if k.strip()}
    if not keys_set:
        return lattice, {}
    before_cell = lattice_to_cell(lattice)
    before_angle_oob = _angle_out_of_range(before_cell, angle_min, angle_max).float()
    before_cond = cond_gram(lattice, pbc_mask=pbc_mask)
    before_vol = torch.linalg.det(lattice).abs().clamp_min(1e-12)
    before_vol_oob = torch.zeros((lattice.shape[0],), device=lattice.device, dtype=torch.float32)
    if "volume" in keys_set and (vol_min is not None or vol_max is not None):
        if vol_min is not None:
            before_vol_oob = torch.where(before_vol < float(vol_min), torch.ones_like(before_vol_oob), before_vol_oob)
        if vol_max is not None:
            before_vol_oob = torch.where(before_vol > float(vol_max), torch.ones_like(before_vol_oob), before_vol_oob)
    before_inplane = inplane_degenerate(
        lattice,
        pbc_mask=pbc_mask,
        a_min=inplane_a_min,
        b_min=inplane_b_min,
        gamma_min=inplane_gamma_min,
        gamma_max=inplane_gamma_max,
        area_min=inplane_area_min,
    ).float()

    lat = lattice
    for _ in range(max(int(max_iters), 1)):
        cell = lattice_to_cell(lat)
        if "angle" in keys_set:
            cell = project_cell_angles_inplace(cell, angle_min=angle_min, angle_max=angle_max)
        if "inplane" in keys_set:
            cell = project_cell_inplane_inplace(
                cell,
                a_min=inplane_a_min,
                b_min=inplane_b_min,
                gamma_min=inplane_gamma_min,
                gamma_max=inplane_gamma_max,
                area_min=inplane_area_min,
            )
        # Rebuild lattice while keeping original orientation as much as possible.
        lat = cell_to_lattice_oriented(cell, ref_lattice=lat)
        if "cond" in keys_set and cond_max is not None:
            lat = project_cond_svd(lat, cond_max=float(cond_max), pbc_mask=pbc_mask)
        # second pass angle/inplane clamp after cond (helps if cond clamp perturbs)
        if ("angle" in keys_set) or ("inplane" in keys_set):
            cell2 = lattice_to_cell(lat)
            if "angle" in keys_set:
                cell2 = project_cell_angles_inplace(cell2, angle_min=angle_min, angle_max=angle_max)
            if "inplane" in keys_set:
                cell2 = project_cell_inplane_inplace(
                    cell2,
                    a_min=inplane_a_min,
                    b_min=inplane_b_min,
                    gamma_min=inplane_gamma_min,
                    gamma_max=inplane_gamma_max,
                    area_min=inplane_area_min,
                )
            lat = cell_to_lattice_oriented(cell2, ref_lattice=lat)

        # Early stop if everything is satisfied
        cell_chk = lattice_to_cell(lat)
        angle_ok = True
        if "angle" in keys_set:
            angle_ok = not bool(_angle_out_of_range(cell_chk, angle_min, angle_max).any().item())
        inplane_ok = True
        if "inplane" in keys_set:
            inplane_ok = not bool(
                inplane_degenerate(
                    lat,
                    pbc_mask=pbc_mask,
                    a_min=inplane_a_min,
                    b_min=inplane_b_min,
                    gamma_min=inplane_gamma_min,
                    gamma_max=inplane_gamma_max,
                    area_min=inplane_area_min,
                )
                .any()
                .item()
            )
        cond_ok = True
        if "cond" in keys_set and cond_max is not None:
            cond_ok = not bool((cond_gram(lat, pbc_mask=pbc_mask) > float(cond_max)).any().item())
        if angle_ok and inplane_ok and cond_ok:
            break

    # Optional: clamp volume into [vol_min, vol_max] by scaling only the in-plane
    # lattice vectors. This is a pragmatic guardrail to prevent bad_volume from
    # dominating failure reasons. If this increases collision rate, compensate
    # via stronger min_dist repulsion (sampling-side), not by disabling clamp.
    vol_scale = torch.ones((lattice.shape[0],), device=lattice.device, dtype=lattice.dtype)
    if "volume" in keys_set and (vol_min is not None or vol_max is not None):
        v = torch.linalg.det(lat).abs().clamp_min(1e-12)
        v_lo = float(vol_min) if vol_min is not None else None
        v_hi = float(vol_max) if vol_max is not None else None
        target = v
        if v_hi is not None:
            target = torch.minimum(target, torch.tensor(v_hi, device=v.device, dtype=v.dtype))
        if v_lo is not None:
            target = torch.maximum(target, torch.tensor(v_lo, device=v.device, dtype=v.dtype))

        # Scaling two in-plane rows changes volume by s^2.
        scale = torch.sqrt((target / v).clamp(min=1e-6, max=1e6))

        # Do not scale down below what would violate the in-plane minimum constraints.
        axes = [i for i, vv in enumerate(pbc_mask) if int(vv) == 1]
        if len(axes) == 2:
            a_vec = lat[:, axes[0], :]
            b_vec = lat[:, axes[1], :]
            a_len = _safe_norm(a_vec)
            b_len = _safe_norm(b_vec)
            area = _safe_norm(torch.cross(a_vec, b_vec, dim=-1))
            s_min = torch.maximum(
                torch.maximum(
                    torch.tensor(float(inplane_a_min), device=v.device, dtype=v.dtype) / a_len,
                    torch.tensor(float(inplane_b_min), device=v.device, dtype=v.dtype) / b_len,
                ),
                torch.sqrt(torch.tensor(float(inplane_area_min), device=v.device, dtype=v.dtype) / area),
            )
            scale = torch.maximum(scale, s_min)

            lat = lat.clone()
            lat[:, axes[0], :] = lat[:, axes[0], :] * scale.view(-1, 1)
            lat[:, axes[1], :] = lat[:, axes[1], :] * scale.view(-1, 1)
            vol_scale = scale
            det = torch.linalg.det(lat)
            flip = det < 0
            if flip.any():
                lat[flip, axes[0], :] = -lat[flip, axes[0], :]

    after_cell = lattice_to_cell(lat)
    after_angle_oob = _angle_out_of_range(after_cell, angle_min, angle_max).float()
    after_cond = cond_gram(lat, pbc_mask=pbc_mask)
    after_vol = torch.linalg.det(lat).abs().clamp_min(1e-12)
    after_vol_oob = torch.zeros((lattice.shape[0],), device=lattice.device, dtype=torch.float32)
    if "volume" in keys_set and (vol_min is not None or vol_max is not None):
        if vol_min is not None:
            after_vol_oob = torch.where(after_vol < float(vol_min), torch.ones_like(after_vol_oob), after_vol_oob)
        if vol_max is not None:
            after_vol_oob = torch.where(after_vol > float(vol_max), torch.ones_like(after_vol_oob), after_vol_oob)
    after_inplane = inplane_degenerate(
        lat,
        pbc_mask=pbc_mask,
        a_min=inplane_a_min,
        b_min=inplane_b_min,
        gamma_min=inplane_gamma_min,
        gamma_max=inplane_gamma_max,
        area_min=inplane_area_min,
    ).float()
    delta = (lat - lattice).reshape(lattice.shape[0], -1)
    base = lattice.reshape(lattice.shape[0], -1)
    delta_norm = torch.linalg.norm(delta, dim=-1) / torch.linalg.norm(base, dim=-1).clamp_min(1e-8)
    trigger_any_bool = (before_angle_oob > 0) | (before_inplane > 0) | (before_vol_oob > 0)
    if cond_max is not None:
        trigger_any_bool = trigger_any_bool | (before_cond > float(cond_max))
    stats = {
        "angle_oob_before": before_angle_oob,
        "angle_oob_after": after_angle_oob,
        "cond_before": before_cond,
        "cond_after": after_cond,
        "vol_before": before_vol,
        "vol_after": after_vol,
        "vol_oob_before": before_vol_oob,
        "vol_oob_after": after_vol_oob,
        "vol_scale_inplane": vol_scale,
        "inplane_degen_before": before_inplane,
        "inplane_degen_after": after_inplane,
        "delta_norm": delta_norm,
        "trigger_any": trigger_any_bool.float(),
    }
    return lat, stats


__all__ = [
    "CellParams",
    "cond_gram",
    "inplane_degenerate",
    "lattice_to_cell",
    "cell_to_lattice_oriented",
    "post_step_project",
    "project_cond_svd",
]
