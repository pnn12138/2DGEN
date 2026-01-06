from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    eps_area: float = 1e-6
    eps_inv: float = 1e-12
    round_prec: float = 1e-6
    tie_break_eps: float = 1e-4
    z_norm_clip: float = 1.5
    cond_max: float = 1e10
    thickness_q_low: float = 0.01
    thickness_q_high: float = 0.99
    max_atomic_number: int = 118
    moment_eps: float = 1e-8


def _wrap01(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def _circular_mean(values: np.ndarray) -> float:
    angles = 2.0 * np.pi * _wrap01(values)
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    return float(np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi))


def _normalize(vec: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def _pca_normal(pos_cart: np.ndarray, fallback_sign: np.ndarray | None) -> np.ndarray:
    centered = pos_cart - np.mean(pos_cart, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n_pca = vh[-1]
    if fallback_sign is not None and np.linalg.norm(fallback_sign) > 0:
        if np.dot(n_pca, fallback_sign) < 0:
            n_pca = -n_pca
    else:
        if n_pca[2] < 0:
            n_pca = -n_pca
    return n_pca


def _enumerate_unimodular() -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            for r in (-1, 0, 1):
                for s in (-1, 0, 1):
                    det = p * s - q * r
                    if abs(det) != 1:
                        continue
                    candidates.append(np.array([[p, q], [r, s]], dtype=np.int64))
    return candidates


_UNIMODULAR_CANDIDATES = _enumerate_unimodular()


def _score_basis(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    a_len = np.linalg.norm(a_vec)
    b_len = np.linalg.norm(b_vec)
    if a_len <= 0 or b_len <= 0:
        return float("inf")
    cos_gamma = float(np.dot(a_vec, b_vec) / (a_len * b_len))
    cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))
    gamma = float(np.degrees(np.arccos(cos_gamma)))
    score = a_len + b_len + 0.1 * abs(cos_gamma)
    if a_len > b_len:
        score += 10.0 + a_len - b_len
    if gamma < 60.0 or gamma > 120.0:
        score += 10.0 + abs(gamma - 90.0) / 90.0
    return score


def _reduce_2d_basis(a_in: np.ndarray, b_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    basis = np.stack([a_in, b_in], axis=1)
    best_score = float("inf")
    best = (a_in, b_in)
    best_u = np.array([[1, 0], [0, 1]], dtype=np.int64)
    for u in _UNIMODULAR_CANDIDATES:
        cand = basis @ u
        a_vec = cand[:, 0]
        b_vec = cand[:, 1]
        score = _score_basis(a_vec, b_vec)
        if score < best_score - 1e-12:
            best_score = score
            best = (a_vec, b_vec)
            best_u = u
        elif abs(score - best_score) <= 1e-12:
            if tuple(u.flatten().tolist()) < tuple(best_u.flatten().tolist()):
                best = (a_vec, b_vec)
                best_u = u
    return best


def _least_squares_uv(a_vec: np.ndarray, b_vec: np.ndarray, r_parallel: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    basis = np.stack([a_vec, b_vec], axis=1)
    ata = basis.T @ basis
    ata = ata + np.eye(2) * cfg.eps_inv
    cond = np.linalg.cond(ata)
    if not np.isfinite(cond) or cond > cfg.cond_max:
        uv, _, _, _ = np.linalg.lstsq(basis, r_parallel.T, rcond=None)
        return uv.T
    inv = np.linalg.solve(ata, basis.T)
    return r_parallel @ inv.T


def preprocess_cartesian(
    cell: np.ndarray, pos_cart: np.ndarray, z_numbers: np.ndarray, cfg: PreprocessConfig
) -> Dict[str, np.ndarray]:
    cell = np.asarray(cell, dtype=np.float64)
    pos_cart = np.asarray(pos_cart, dtype=np.float64)
    z_numbers = np.asarray(z_numbers, dtype=np.int64)
    if pos_cart.shape[0] != z_numbers.shape[0]:
        raise ValueError("pos_cart and z_numbers must have the same length.")

    a_vec = cell[0]
    b_vec = cell[1]
    c_vec = cell[2]

    n_raw = np.cross(a_vec, b_vec)
    n_norm = np.linalg.norm(n_raw)
    if n_norm < cfg.eps_area:
        n_vec = _normalize(_pca_normal(pos_cart, None), cfg.eps_area)
    else:
        n_vec = _normalize(n_raw, cfg.eps_area)

    z = pos_cart @ n_vec
    r_parallel = pos_cart - z[:, None] * n_vec[None, :]

    a_in = a_vec - np.dot(a_vec, n_vec) * n_vec
    b_in = b_vec - np.dot(b_vec, n_vec) * n_vec

    if n_norm < cfg.eps_area:
        n_vec = _normalize(_pca_normal(pos_cart, np.cross(a_in, b_in)), cfg.eps_area)
        a_in = a_vec - np.dot(a_vec, n_vec) * n_vec
        b_in = b_vec - np.dot(b_vec, n_vec) * n_vec

    a_hat, b_hat = _reduce_2d_basis(a_in, b_in)
    if np.dot(np.cross(a_hat, b_hat), n_vec) < 0:
        b_hat = -b_hat

    uv = _least_squares_uv(a_hat, b_hat, r_parallel, cfg)
    u = _wrap01(uv[:, 0])
    v = _wrap01(uv[:, 1])

    u_shift = _circular_mean(u)
    v_shift = _circular_mean(v)
    u = _wrap01(u - u_shift)
    v = _wrap01(v - v_shift)

    z = z - np.mean(z)
    m1 = float(np.sum(z_numbers * z))
    if m1 < -cfg.moment_eps:
        z = -z
        n_vec = -n_vec
    elif abs(m1) <= cfg.moment_eps:
        m3 = float(np.sum(z_numbers * (z**3)))
        if m3 < 0:
            z = -z
            n_vec = -n_vec

    t = float(np.quantile(z, cfg.thickness_q_high) - np.quantile(z, cfg.thickness_q_low))
    if t < cfg.eps_inv:
        t = float(np.ptp(z))
    if t < cfg.eps_inv:
        t = 1.0
    z_norm = z / (t + cfg.eps_inv)
    z_norm = np.clip(z_norm, -cfg.z_norm_clip, cfg.z_norm_clip)

    g2d = np.array(
        [[np.dot(a_hat, a_hat), np.dot(a_hat, b_hat)], [np.dot(a_hat, b_hat), np.dot(b_hat, b_hat)]],
        dtype=np.float64,
    )
    det_g = float(np.linalg.det(g2d))
    area = float(np.sqrt(max(det_g, cfg.eps_inv)))
    log_area = float(np.log(area))
    g_shape = g2d / area
    try:
        chol = np.linalg.cholesky(g_shape + np.eye(2) * cfg.eps_inv)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(g_shape + np.eye(2) * (cfg.eps_inv * 10.0))
    p1 = float(np.log(max(chol[0, 0], cfg.eps_inv)))
    p2 = float(chol[1, 0])
    lattice_param = np.array([log_area, p1, p2], dtype=np.float32)

    z_key = np.round(z_norm / cfg.round_prec) * cfg.round_prec
    u_key = np.round(u / cfg.round_prec) * cfg.round_prec
    v_key = np.round(v / cfg.round_prec) * cfg.round_prec
    original_idx = np.arange(len(z_numbers))
    tie_break = cfg.round_prec * cfg.tie_break_eps
    if tie_break > 0:
        z_key = z_key + original_idx * tie_break
        u_key = u_key + original_idx * tie_break
        v_key = v_key + original_idx * tie_break
    order_idx = np.lexsort((original_idx, v_key, u_key, z_key, z_numbers))

    z_sorted = z_numbers[order_idx]
    u_sorted = u[order_idx]
    v_sorted = v[order_idx]
    z_norm_sorted = z_norm[order_idx]
    uv_angle = np.stack(
        [
            np.cos(2.0 * np.pi * u_sorted),
            np.sin(2.0 * np.pi * u_sorted),
            np.cos(2.0 * np.pi * v_sorted),
            np.sin(2.0 * np.pi * v_sorted),
        ],
        axis=-1,
    ).astype(np.float32)

    counts = np.zeros((cfg.max_atomic_number,), dtype=np.int64)
    for z_val in z_sorted:
        if 1 <= int(z_val) <= cfg.max_atomic_number:
            counts[int(z_val) - 1] += 1

    c_len = float(abs(np.dot(c_vec, n_vec)))
    if c_len < cfg.eps_inv:
        c_len = float(np.linalg.norm(c_vec))
    if c_len < cfg.eps_inv:
        c_len = 1.0
    c_hat = n_vec * c_len
    lattice_canon = np.stack([a_hat, b_hat, c_hat], axis=0).astype(np.float32)

    return {
        "Z": z_sorted.astype(np.int64),
        "u": u_sorted.astype(np.float32),
        "v": v_sorted.astype(np.float32),
        "z_norm": z_norm_sorted.astype(np.float32),
        "uv_angle": uv_angle,
        "t": np.array(t, dtype=np.float32),
        "a_hat": a_hat.astype(np.float32),
        "b_hat": b_hat.astype(np.float32),
        "n": n_vec.astype(np.float32),
        "lattice_canon": lattice_canon,
        "u_shift": np.array(u_shift, dtype=np.float32),
        "v_shift": np.array(v_shift, dtype=np.float32),
        "lattice_param": lattice_param.astype(np.float32),
        "counts_vector": counts.astype(np.int64),
        "order_idx": order_idx.astype(np.int64),
    }


__all__ = ["PreprocessConfig", "preprocess_cartesian"]
