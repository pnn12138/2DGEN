from __future__ import annotations

import argparse
import json
import shlex
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Element

from twodgen.common.geometry_np import choose_vacuum_axis, min_dist_and_shifts, thickness_vacuum
from twodgen.evaluate.cache import load_eval_cache


EVAL_SCHEMA_VERSION = "eval_samples_v1"
VALID_CRITERIA = [
    "n_atoms >= 3",
    "non-empty atoms",
    "volume within [v_min, v_max] when provided",
    "determinant > 0",
    "lattice Gram matrix is SPD",
    "condition number <= cond_max when provided (in-plane Gram for 2D slabs)",
    "min_dist >= min_dist_cut (exact MIC under pbc_mask)",
    "dup_ratio <= 0.2 (grid-quantized with dup_eps)",
    "angles alpha/beta/gamma within [30, 150] degrees",
]


_ELEMENT_SYMBOLS = [None] + [Element.from_Z(z).symbol for z in range(1, 119)]
_SPACEGROUP_MAX = 230

# Fail reason priority used to choose a single "main" geom failure.
_FAIL_REASON_GEOM_PRIORITY = [
    "collision",
    "angle_out_of_range",
    "inplane_degenerate",
    "cond_overflow",
    "bad_volume",
    "non_spd",
    "det_nonpos",
    "duplicate_coord",
    "low_atoms",
    "empty_atoms",
]


def _main_fail_reason(reasons: List[str], priority: List[str]) -> str:
    if not reasons:
        return ""
    seen = set(reasons)
    for key in priority:
        if key in seen:
            return key
    return reasons[0]


def _inplane_metrics(
    lattice: np.ndarray,
    *,
    pbc_mask: Tuple[int, int, int],
) -> Tuple[float, float, float, float]:
    axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
    if len(axes) != 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a_vec = lattice[axes[0]]
    b_vec = lattice[axes[1]]
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    denom = max(a_len * b_len, 1e-12)
    cos_g = float(np.dot(a_vec, b_vec) / denom)
    cos_g = float(np.clip(cos_g, -1.0, 1.0))
    gamma = float(np.degrees(np.arccos(cos_g)))
    area = float(np.linalg.norm(np.cross(a_vec, b_vec)))
    return a_len, b_len, gamma, area


def _cond_gram_inplane(lattice: np.ndarray, *, pbc_mask: Tuple[int, int, int]) -> float:
    """
    In-plane Gram condition number for 2D slabs (two periodic axes).
    Returns inf if the in-plane Gram is not SPD / numeric.
    """
    axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
    if len(axes) != 2:
        return float("nan")
    sub = lattice[axes]  # (2,3) row-basis
    gram = sub @ sub.T  # (2,2)
    try:
        eigvals = np.linalg.eigvalsh(gram)
    except Exception:
        return float("inf")
    if not np.all(np.isfinite(eigvals)) or np.any(eigvals <= 0.0):
        return float("inf")
    return float(eigvals.max() / max(float(eigvals.min()), 1e-12))


def _spacegroup_number(
    lattice: np.ndarray, frac: np.ndarray, numbers: np.ndarray, symprec: float
) -> Optional[int]:
    try:
        import spglib  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        cell = (lattice, frac, numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=float(symprec))
        if dataset is None:
            return None
        return int(dataset.get("number"))
    except Exception:
        return None


def _load_element_refs(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def _atomic_ref_map(
    element_refs: Optional[Dict[str, float]],
    default_mu: float,
) -> list[float]:
    ref_map: list[float] = [default_mu]
    for symbol in _ELEMENT_SYMBOLS[1:]:
        if element_refs is None:
            ref_map.append(float(default_mu))
        else:
            ref_map.append(float(element_refs.get(symbol, default_mu)))
    return ref_map


def _formation_energy_per_atom(
    total_energy: Optional[float],
    counts_vec: np.ndarray,
    atomic_ref_map: list[float],
) -> Optional[float]:
    if total_energy is None or not np.isfinite(total_energy):
        return None
    n_atoms = int(np.sum(counts_vec))
    if n_atoms <= 0:
        return None
    ref_energy = 0.0
    for idx, count in enumerate(counts_vec):
        if count <= 0:
            continue
        z = idx + 1
        ref_energy += atomic_ref_map[z] * float(count)
    if not np.isfinite(ref_energy):
        ref_energy = float(atomic_ref_map[0] * n_atoms)
    return float((total_energy - ref_energy) / max(n_atoms, 1))


def build_eval_params(
    *,
    min_dist_cut: float,
    bond_cut: float,
    dup_eps: float,
    vacuum_min: Optional[float],
    v_min: Optional[float],
    v_max: Optional[float],
    pbc_mask: Tuple[int, int, int],
    formation_energy_threshold: float,
    element_refs_path: Optional[Path],
    target_spacegroup: Optional[int] = None,
    spacegroup_symprec: float = 1e-2,
    cond_max: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "min_dist_cut": float(min_dist_cut),
        "bond_cut": float(bond_cut),
        "dup_eps": float(dup_eps),
        "vacuum_min": float(vacuum_min) if vacuum_min is not None else None,
        "v_min": float(v_min) if v_min is not None else None,
        "v_max": float(v_max) if v_max is not None else None,
        "pbc_mask": pbc_mask,
        "formation_energy_threshold": float(formation_energy_threshold),
        "element_refs_path": str(element_refs_path) if element_refs_path is not None else None,
        "target_spacegroup": int(target_spacegroup) if target_spacegroup is not None else None,
        "spacegroup_symprec": float(spacegroup_symprec),
        "cond_max": float(cond_max) if cond_max is not None else None,
    }


def write_eval_outputs(
    *,
    out_dir: Path,
    per_sample: List[Dict[str, Any]],
    tier0: Dict[str, Any],
    tier1: Dict[str, Any],
    eval_params: Dict[str, Any],
    success_manifest: Optional[List[Dict[str, Any]]] = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tier0.setdefault("schema_version", EVAL_SCHEMA_VERSION)
    tier0.setdefault("valid_criteria", list(VALID_CRITERIA))
    tier0["eval_params"] = eval_params
    if run_context is not None:
        tier0["run_context"] = run_context

    per_sample_path = out_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    typo_path = out_dir / "per_sanmple.jsonl"
    if typo_path.exists():
        typo_backup = out_dir / "per_sanmple.jsonl.bak"
        typo_path.replace(typo_backup)
    with (out_dir / "tier0_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(tier0, f, indent=2, ensure_ascii=True)
    with (out_dir / "tier1_2d_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(tier1, f, indent=2, ensure_ascii=True)
    if success_manifest:
        with (out_dir / "success_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(success_manifest, f, indent=2, ensure_ascii=True)


def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def _load_npz_stats(npz_path: Path) -> Optional[Tuple[float, float]]:
    data = np.load(npz_path)
    lattice = data["lattice"] if "lattice" in data else None
    if lattice is None:
        return None
    vols = np.abs(np.linalg.det(lattice))
    v_min = float(np.percentile(vols, 1.0))
    v_max = float(np.percentile(vols, 99.0))
    return v_min, v_max


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    arr = np.asarray(clean, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _counts_from_samples(z: np.ndarray, atom_mask: np.ndarray, num_elements: int) -> np.ndarray:
    counts = np.zeros((z.shape[0], num_elements), dtype=np.int64)
    valid = (atom_mask > 0.5) & (z > 0)
    batch_idx, atom_idx = np.where(valid)
    if batch_idx.size == 0:
        return counts
    elem_idx = z[batch_idx, atom_idx].astype(np.int64) - 1
    keep = (elem_idx >= 0) & (elem_idx < num_elements)
    if keep.any():
        np.add.at(counts, (batch_idx[keep], elem_idx[keep]), 1)
    return counts


def _gcc_ratio(n_atoms: int, edges: List[Tuple[int, int]]) -> float:
    if n_atoms == 0:
        return 0.0
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    visited = [False] * n_atoms
    max_size = 0
    for i in range(n_atoms):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        max_size = max(max_size, size)
    return float(max_size / n_atoms)


def _eval_samples(
    samples: Dict[str, np.ndarray],
    v_min: Optional[float],
    v_max: Optional[float],
    min_dist_cut: float,
    bond_cut: float,
    dup_eps: float,
    pbc_mask: Tuple[int, int, int],
    vacuum_min: Optional[float] = None,
    atomic_ref_map: Optional[list[float]] = None,
    formation_energy_threshold: float = 0.0,
    success_top_k: int = 10,
    target_spacegroup: Optional[int] = None,
    spacegroup_symprec: float = 1e-2,
    cond_max: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    z = samples["z"]
    frac = samples["frac"]
    lattice = samples["lattice"]
    atom_mask = samples["atom_mask"]
    cond_counts_vector = samples.get("cond_counts_vector")
    cond_source = samples.get("cond_counts_source")
    cond_source_value = None
    if cond_source is not None:
        try:
            cond_source_value = str(cond_source.item())
        except Exception:
            cond_source_value = str(cond_source)
    cond_metrics = None
    cond_match_suspect = False
    exact_match = np.array([], dtype=bool)
    l1 = np.array([], dtype=float)
    if cond_counts_vector is not None:
        num_elements = int(cond_counts_vector.shape[1])
        gen_counts = _counts_from_samples(z, atom_mask, num_elements=num_elements)
        cond_counts = cond_counts_vector.astype(np.int64)
        exact_match = np.all(gen_counts == cond_counts, axis=1)
        l1 = np.sum(np.abs(gen_counts - cond_counts), axis=1).astype(float)
        total_cond = np.sum(cond_counts, axis=1).astype(float)
        total_gen = np.sum(gen_counts, axis=1).astype(float)
        l1_norm = l1 / np.maximum(total_cond, 1.0)
        comp_l1 = np.full_like(l1, np.nan, dtype=float)
        comp_cos = np.full_like(l1, np.nan, dtype=float)
        valid_comp = (total_cond > 0) & (total_gen > 0)
        if np.any(valid_comp):
            gen_frac = gen_counts[valid_comp] / total_gen[valid_comp][:, None]
            cond_frac = cond_counts[valid_comp] / total_cond[valid_comp][:, None]
            comp_l1[valid_comp] = np.sum(np.abs(gen_frac - cond_frac), axis=1)
            dot = np.sum(gen_frac * cond_frac, axis=1)
            norm = np.linalg.norm(gen_frac, axis=1) * np.linalg.norm(cond_frac, axis=1)
            comp_cos[valid_comp] = dot / np.maximum(norm, 1e-12)
        cond_metrics = {
            "exact_match": exact_match,
            "l1": l1,
            "l1_norm": l1_norm,
            "comp_l1": comp_l1,
            "comp_cos": comp_cos,
        }
    if (
        cond_metrics is not None
        and cond_source_value is None
        and exact_match.size
        and np.all(exact_match)
        and np.all(l1 == 0)
    ):
        cond_match_suspect = True

    per_sample: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}
    elem_counts: Dict[str, int] = {}

    min_dists: List[float] = []
    collision_min_dists: List[float] = []
    volumes: List[float] = []
    conds: List[float] = []
    conds_full: List[float] = []
    cond_overflow_flags: List[int] = []
    n_atoms_list: List[int] = []
    angles_alpha: List[float] = []
    angles_beta: List[float] = []
    angles_gamma: List[float] = []
    angle_out_flags: List[int] = []
    inplane_degen_flags: List[int] = []
    inplane_area_stats: List[float] = []
    inplane_gamma_stats: List[float] = []
    same_elem_min_dists: List[float] = []

    thicknesses: List[float] = []
    vacuums: List[float] = []
    cross_vacuum_flags: List[int] = []
    vacuum_ok_flags: List[int] = []
    gcc_ratios: List[float] = []
    anisotropies: List[float] = []

    valid_flags: List[int] = []
    valid_2d_flags: List[int] = []

    counts_batch = _counts_from_samples(z, atom_mask, num_elements=len(_ELEMENT_SYMBOLS) - 1)
    project_cond_before = samples.get("project_cond_before")
    project_cond_after = samples.get("project_cond_after")
    project_delta_cond = samples.get("project_delta_cond")
    project_angle_out_before = samples.get("project_angle_out_before")
    project_angle_out_after = samples.get("project_angle_out_after")
    project_trigger = samples.get("project_trigger")
    post_project_trigger_any = samples.get("post_project_trigger_any")
    post_project_delta_norm = samples.get("post_project_delta_norm")
    post_project_vol_before = samples.get("post_project_vol_before")
    post_project_vol_after = samples.get("post_project_vol_after")
    post_project_vol_scale_inplane = samples.get("post_project_vol_scale_inplane")
    energy_mlip_values = samples.get("energy_mlip")
    relaxed_flag_values = samples.get("relaxed_flag")
    min_dist_relax_values = samples.get("min_dist_relax")
    energy_stats: List[float] = []
    formation_stats: List[float] = []
    success_flags: List[int] = []
    success_geom_flags: List[int] = []
    success_energy_flags: List[int] = []
    energy_available_flags: List[int] = []
    energy_skipped_geom_flags: List[int] = []
    energy_skipped_mlip_flags: List[int] = []
    energy_fail_reason_counts: Dict[str, int] = {}
    relaxed_flags: List[int] = []
    min_dist_relax_stats: List[float] = []
    sg_numbers: List[Optional[int]] = []
    sg_match_flags: List[int] = []
    sg_violation_flags: List[int] = []
    cond_spacegroup = samples.get("cond_spacegroup")
    if cond_spacegroup is not None:
        cond_spacegroup = cond_spacegroup.reshape(-1)

    for i in range(z.shape[0]):
        mask = atom_mask[i] > 0.5
        z_i = z[i][mask].astype(int)
        frac_i = frac[i][mask]
        lattice_i = lattice[i]
        n_atoms = int(mask.sum())

        reasons: List[str] = []
        if n_atoms == 0:
            reasons.append("empty_atoms")
        if n_atoms < 3:
            reasons.append("low_atoms")

        det = float(np.linalg.det(lattice_i))
        vol = float(abs(det))
        if not np.isfinite(det) or det <= 0.0:
            reasons.append("det_nonpos")
        if v_min is not None and v_max is not None:
            if vol < v_min or vol > v_max:
                reasons.append("bad_volume")

        gram = lattice_i @ lattice_i.T
        eigvals = np.linalg.eigvalsh(gram)
        if not np.all(np.isfinite(eigvals)) or np.any(eigvals <= 0.0):
            reasons.append("non_spd")
            cond_full = float("inf")
        else:
            cond_full = float(eigvals.max() / max(eigvals.min(), 1e-12))

        # For 2D slabs, use in-plane Gram cond for cond_max checks.
        axes = [i for i, v in enumerate(pbc_mask) if int(v) == 1]
        if len(axes) == 2:
            cond = _cond_gram_inplane(lattice_i, pbc_mask=pbc_mask)
        else:
            cond = cond_full

        cond_lattice = float(math.sqrt(cond)) if np.isfinite(cond) and cond >= 0 else float("nan")
        if cond_max is not None and np.isfinite(cond) and cond > float(cond_max):
            reasons.append("cond_overflow")
            cond_overflow_flags.append(1)
        else:
            cond_overflow_flags.append(0)

        if n_atoms > 0:
            min_dist, dist, shifts = min_dist_and_shifts(frac_i, lattice_i, pbc_mask=pbc_mask)
            min_dists.append(min_dist)
            if min_dist < min_dist_cut:
                reasons.append("collision")
                collision_min_dists.append(min_dist)
            if n_atoms > 1:
                unique, counts = np.unique(z_i, return_counts=True)
                if np.any(counts >= 2):
                    same_mask = z_i[:, None] == z_i[None, :]
                    same_dist = np.where(same_mask, dist, np.inf)
                    same_elem_min_dists.append(float(np.min(same_dist)))
                else:
                    same_elem_min_dists.append(None)
            else:
                same_elem_min_dists.append(None)

            quant = np.round(frac_i / dup_eps)
            uniq = np.unique(quant, axis=0)
            dup_ratio = 1.0 - (len(uniq) / n_atoms)
            if dup_ratio > 0.2:
                reasons.append("duplicate_coord")
        else:
            min_dist = float("inf")
            dist = np.zeros((0, 0))
            shifts = np.zeros((0, 0, 3))
            dup_ratio = float("nan")
            same_elem_min_dists.append(None)

        lengths = np.linalg.norm(lattice_i, axis=1)
        if np.all(lengths > 0):
            a_vec, b_vec, c_vec = lattice_i
            cos_alpha = float(np.dot(b_vec, c_vec) / (np.linalg.norm(b_vec) * np.linalg.norm(c_vec)))
            cos_beta = float(np.dot(a_vec, c_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(c_vec)))
            cos_gamma = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
            cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
            cos_beta = float(np.clip(cos_beta, -1.0, 1.0))
            cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))
            alpha = float(np.degrees(np.arccos(cos_alpha)))
            beta = float(np.degrees(np.arccos(cos_beta)))
            gamma = float(np.degrees(np.arccos(cos_gamma)))
            angles_alpha.append(alpha)
            angles_beta.append(beta)
            angles_gamma.append(gamma)
            angle_out = int((alpha < 30.0 or alpha > 150.0) or (beta < 30.0 or beta > 150.0) or (gamma < 30.0 or gamma > 150.0))
            angle_out_flags.append(angle_out)
            if angle_out:
                reasons.append("angle_out_of_range")
        else:
            angles_alpha.append(float("nan"))
            angles_beta.append(float("nan"))
            angles_gamma.append(float("nan"))
            angle_out_flags.append(0)

        # in-plane degeneracy (2D guardrail) based on pbc_mask periodic axes
        a_len_in, b_len_in, gamma_in, area_in = _inplane_metrics(lattice_i, pbc_mask=pbc_mask)
        inplane_gamma_stats.append(gamma_in)
        inplane_area_stats.append(area_in)
        inplane_degen = int(
            (not np.isfinite(a_len_in))
            or (a_len_in < 2.0)
            or (not np.isfinite(b_len_in))
            or (b_len_in < 2.0)
            or (not np.isfinite(gamma_in))
            or (gamma_in < 30.0)
            or (gamma_in > 150.0)
            or (not np.isfinite(area_in))
            or (area_in < 4.0)
        )
        inplane_degen_flags.append(inplane_degen)
        if inplane_degen:
            reasons.append("inplane_degenerate")
        c_idx, c_len, _ = choose_vacuum_axis(lattice_i)
        if n_atoms > 0:
            thickness, vacuum = thickness_vacuum(frac_i[:, c_idx], c_len)
        else:
            thickness, vacuum = float("nan"), float("nan")

        cross_vacuum = False
        edges: List[Tuple[int, int]] = []
        if n_atoms > 1:
            dist_3d, shifts_3d = None, None
            if pbc_mask[c_idx] == 0:
                _, dist_3d, shifts_3d = min_dist_and_shifts(frac_i, lattice_i, pbc_mask=(1, 1, 1))
            for a in range(n_atoms):
                for b in range(a + 1, n_atoms):
                    if dist[a, b] < bond_cut:
                        edges.append((a, b))
                    if dist_3d is not None and dist_3d[a, b] < bond_cut:
                        n_c = shifts_3d[a, b, c_idx]
                        if n_c != 0:
                            cross_vacuum = True
        gcc_ratio = _gcc_ratio(n_atoms, edges)

        anisotropy = float(c_len / max(np.mean([l for j, l in enumerate(lengths) if j != c_idx]), 1e-8))

        valid = len(reasons) == 0
        valid_2d = valid and (not cross_vacuum)

        for reason in reasons:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

        if valid:
            for z_val in z_i.tolist():
                if z_val > 0:
                    key = str(z_val)
                    elem_counts[key] = elem_counts.get(key, 0) + 1

        volumes.append(vol)
        conds.append(cond)
        conds_full.append(cond_full)
        n_atoms_list.append(n_atoms)
        thicknesses.append(thickness)
        vacuums.append(vacuum)
        cross_vacuum_flags.append(int(cross_vacuum))
        if vacuum_min is not None and np.isfinite(vacuum):
            vacuum_ok_flags.append(int(float(vacuum) >= float(vacuum_min)))
        gcc_ratios.append(gcc_ratio)
        anisotropies.append(anisotropy)
        valid_flags.append(int(valid))
        valid_2d_flags.append(int(valid_2d))

        counts_vec = counts_batch[i]
        energy_val: Optional[float] = None
        if energy_mlip_values is not None:
            energy_raw = energy_mlip_values[i]
            if np.isfinite(float(energy_raw)):
                energy_val = float(energy_raw)
        relaxed_flag_value = None
        if relaxed_flag_values is not None:
            try:
                relaxed_flag_value = int(relaxed_flag_values[i])
            except Exception:
                relaxed_flag_value = 0
            relaxed_flags.append(relaxed_flag_value)
        min_dist_relax_value = None
        if min_dist_relax_values is not None:
            try:
                min_dist_relax_val = float(min_dist_relax_values[i])
                if np.isfinite(min_dist_relax_val):
                    min_dist_relax_value = min_dist_relax_val
                    min_dist_relax_stats.append(min_dist_relax_val)
            except Exception:
                pass
        formation_energy: Optional[float] = None
        if atomic_ref_map is not None:
            formation_energy = _formation_energy_per_atom(
                energy_val, counts_vec, atomic_ref_map
            )
        cond_ok = True
        if cond_metrics is not None:
            cond_ok = bool(cond_metrics["exact_match"][i])

        target_sg = target_spacegroup
        if cond_spacegroup is not None:
            if np.size(cond_spacegroup) > i:
                sg_val = int(cond_spacegroup[i])
                if sg_val > 0:
                    target_sg = sg_val
        sg_number = None
        sg_match = None
        if n_atoms > 0 and target_sg is not None:
            sg_number = _spacegroup_number(lattice_i, frac_i, z_i, symprec=spacegroup_symprec)
            sg_match = bool(sg_number == int(target_sg))
        elif n_atoms > 0:
            sg_number = _spacegroup_number(lattice_i, frac_i, z_i, symprec=spacegroup_symprec)
        symmetry_violation = None
        if target_sg is not None:
            symmetry_violation = not bool(sg_match) if sg_match is not None else True
        if sg_match is not None:
            sg_match_flags.append(int(sg_match))
        if symmetry_violation is not None:
            sg_violation_flags.append(int(symmetry_violation))
        sg_numbers.append(sg_number)

        # Energy taxonomy:
        # - energy_available: whether MLIP energy exists (independent of element refs)
        # - success_energy: relax succeeded + numeric sanity (+ optional formation energy threshold)
        energy_available = energy_val is not None
        low_energy = (formation_energy is None) or (formation_energy <= float(formation_energy_threshold))
        symmetry_ok = True if target_sg is None else bool(sg_match)
        success_geom = bool(valid and valid_2d and cond_ok and symmetry_ok)
        energy_skipped_reason = None
        fail_reason_energy = None
        if not success_geom:
            energy_skipped_reason = "geom_fail"
            energy_skipped_geom_flags.append(1)
            energy_skipped_mlip_flags.append(0)
            success_energy = None
        elif not energy_available:
            energy_skipped_reason = "mlip_unavailable"
            energy_skipped_geom_flags.append(0)
            energy_skipped_mlip_flags.append(1)
            success_energy = None
            fail_reason_energy = "missing"
        else:
            energy_skipped_geom_flags.append(0)
            energy_skipped_mlip_flags.append(0)
            ok_relax = bool(relaxed_flag_value) if relaxed_flag_value is not None else True
            if not np.isfinite(float(energy_val)):
                ok_relax = False
                fail_reason_energy = "nan_energy"
            if not ok_relax:
                success_energy = False
                fail_reason_energy = fail_reason_energy or "non_converge"
            else:
                success_energy = bool(low_energy)
                if not success_energy:
                    fail_reason_energy = "high_energy"
        success_flag = bool(success_geom and (bool(success_energy) if success_energy is not None else True))
        if fail_reason_energy is not None:
            energy_fail_reason_counts[fail_reason_energy] = energy_fail_reason_counts.get(fail_reason_energy, 0) + 1
        success_reasons: List[str] = []
        if not success_flag:
            if not valid:
                success_reasons.append("invalid")
            if not valid_2d:
                success_reasons.append("invalid_2d")
            if not cond_ok:
                success_reasons.append("cond_mismatch")
            if not symmetry_ok:
                success_reasons.append("spacegroup_mismatch")
            if energy_available and not low_energy:
                success_reasons.append("high_energy")

        if energy_val is not None:
            energy_stats.append(energy_val)
        if formation_energy is not None:
            formation_stats.append(formation_energy)
        success_flags.append(int(success_flag))
        success_geom_flags.append(int(success_geom))
        if success_energy is not None:
            success_energy_flags.append(int(bool(success_energy)))
        energy_available_flags.append(int(energy_available))

        row = {
            "id": int(i),
            "n_atoms": n_atoms,
            "volume": vol,
            "cond": cond,
            "min_dist": min_dist,
            "min_dist_same_elem": same_elem_min_dists[-1],
            "dup_ratio": dup_ratio,
            "thickness": thickness,
            "vacuum": vacuum,
            "cross_vacuum_bond": cross_vacuum,
            "gcc_ratio": gcc_ratio,
            "anisotropy": anisotropy,
            "valid": valid,
            "valid_2d": valid_2d,
            "fail_reason": "+".join(reasons) if reasons else "",
            "energy_mlip": energy_val,
            "formation_energy_per_atom": formation_energy,
            "relaxed_flag": relaxed_flag_value,
            "min_dist_relax": min_dist_relax_value,
            "success": success_flag,
            "success_geom": success_geom,
            "success_energy": success_energy,
            "energy_available": energy_available,
            "energy_skipped_reason": energy_skipped_reason,
            "fail_reason_energy": fail_reason_energy,
            "success_fail_reason": "+".join(success_reasons) if success_reasons else "",
            "spacegroup_number": sg_number,
            "spacegroup_match": sg_match if sg_match is not None else None,
            "symmetry_violation": symmetry_violation,
        }
        row["cond_gram"] = cond
        row["cond_gram_full"] = cond_full
        row["cond_lattice"] = cond_lattice
        row["inplane_gamma"] = gamma_in
        row["inplane_area"] = area_in
        row["fail_reason_geom"] = _main_fail_reason(reasons, _FAIL_REASON_GEOM_PRIORITY)
        if cond_metrics is not None:
            row.update(
                {
                    "cond_exact_match": bool(cond_metrics["exact_match"][i]),
                    "cond_l1_count_error": float(cond_metrics["l1"][i]),
                    "cond_l1_count_error_norm": float(cond_metrics["l1_norm"][i]),
                    "cond_comp_l1": float(cond_metrics["comp_l1"][i]),
                    "cond_comp_cosine": float(cond_metrics["comp_cos"][i]),
                }
            )
        per_sample.append(row)

    eval_valid_rate = float(np.mean(valid_flags)) if valid_flags else 0.0
    tier0 = {
        "valid_rate_eval": eval_valid_rate,
        "fail_reason_counts": fail_counts,
        "min_dist": _summary_stats(min_dists),
        "min_dist_collision": _summary_stats(collision_min_dists),
        "min_dist_same_elem": _summary_stats(same_elem_min_dists),
        "volume": _summary_stats(volumes),
        "cond": _summary_stats(conds),  # Gram cond (eig ratio); in-plane for 2D slabs
        "cond_full": _summary_stats(conds_full),  # full 3x3 Gram cond (debug)
        "cond_lattice": _summary_stats([float(math.sqrt(c)) for c in conds if c is not None and np.isfinite(c) and c >= 0]),
        "spd_rate": float(np.mean([c < float("inf") for c in conds_full])) if conds_full else 0.0,
        "energy_mlip": _summary_stats(energy_stats),
        "formation_energy_per_atom": _summary_stats(formation_stats),
        "relaxed_rate": float(np.mean(relaxed_flags)) if relaxed_flags else None,
        "min_dist_relax": _summary_stats(min_dist_relax_stats),
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "success_geom_rate": float(np.mean(success_geom_flags)) if success_geom_flags else 0.0,
        "success_energy_rate": float(np.mean(success_energy_flags)) if success_energy_flags else None,
        "energy_available_rate": float(np.mean(energy_available_flags)) if energy_available_flags else None,
        "energy_skipped_geom_rate": float(np.mean(energy_skipped_geom_flags)) if energy_skipped_geom_flags else None,
        "energy_skipped_mlip_rate": float(np.mean(energy_skipped_mlip_flags)) if energy_skipped_mlip_flags else None,
        "fail_reason_energy_counts": energy_fail_reason_counts,
        "spacegroup_match_rate": float(np.mean(sg_match_flags)) if sg_match_flags else None,
        "symmetry_violation_rate": float(np.mean(sg_violation_flags)) if sg_violation_flags else None,
        "angle_alpha": _summary_stats(angles_alpha),
        "angle_beta": _summary_stats(angles_beta),
        "angle_gamma": _summary_stats(angles_gamma),
        "angle_out_of_range_rate": float(np.mean(angle_out_flags)) if angle_out_flags else 0.0,
        "inplane_degen_rate": float(np.mean(inplane_degen_flags)) if inplane_degen_flags else 0.0,
        "inplane_area": _summary_stats(inplane_area_stats),
        "inplane_gamma": _summary_stats(inplane_gamma_stats),
        "n_atoms": _summary_stats(n_atoms_list),
        "element_counts": elem_counts,
        "total_samples": int(z.shape[0]),
    }
    if fail_counts:
        top3 = sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        tier0["fail_reason_top3"] = top3
    if cond_max is not None and cond_overflow_flags:
        tier0["cond_violation_rate"] = float(np.mean(cond_overflow_flags))
        tier0["cond_lattice_violation_rate"] = float(
            np.mean([int(np.isfinite(c) and c > float(cond_max)) for c in conds])
        )
    if project_trigger is not None:
        tier0["project_trigger_rate"] = float(np.mean(project_trigger))
    if project_delta_cond is not None:
        tier0["project_delta_cond"] = _summary_stats(project_delta_cond.tolist())
    if project_angle_out_before is not None and project_angle_out_after is not None:
        tier0["project_angle_out_rate_before"] = float(np.mean(project_angle_out_before))
        tier0["project_angle_out_rate_after"] = float(np.mean(project_angle_out_after))
    if post_project_trigger_any is not None:
        tier0["post_project_trigger_any_rate"] = float(np.mean(post_project_trigger_any))
    if post_project_delta_norm is not None:
        try:
            tier0["post_project_delta_norm"] = _summary_stats(post_project_delta_norm.tolist())
        except Exception:
            pass
    if post_project_vol_scale_inplane is not None:
        try:
            tier0["post_project_vol_scale_inplane"] = _summary_stats(post_project_vol_scale_inplane.tolist())
            tier0["post_project_vol_scaled_rate"] = float(np.mean(np.abs(post_project_vol_scale_inplane - 1.0) > 1e-6))
        except Exception:
            pass
    if post_project_vol_before is not None and post_project_vol_after is not None:
        try:
            tier0["post_project_vol_before"] = _summary_stats(post_project_vol_before.tolist())
            tier0["post_project_vol_after"] = _summary_stats(post_project_vol_after.tolist())
        except Exception:
            pass
    if cond_metrics is not None:
        tier0["cond_match"] = {
            "exact_match_rate": float(np.mean(cond_metrics["exact_match"]))
            if cond_metrics["exact_match"].size
            else 0.0,
            "l1_count_error": _summary_stats(cond_metrics["l1"].tolist()),
            "l1_count_error_norm": _summary_stats(cond_metrics["l1_norm"].tolist()),
            "comp_l1": _summary_stats(cond_metrics["comp_l1"].tolist()),
            "comp_cosine": _summary_stats(cond_metrics["comp_cos"].tolist()),
            "source": cond_source_value,
            "suspect_all_match": bool(cond_match_suspect),
        }
    tier1 = {
        "valid_2d_rate": float(np.mean(valid_2d_flags)) if valid_2d_flags else 0.0,
        "thickness": _summary_stats(thicknesses),
        "vacuum": _summary_stats(vacuums),
        "vacuum_ok_rate": float(np.mean(vacuum_ok_flags)) if vacuum_ok_flags else None,
        "cross_vacuum_rate": float(np.mean(cross_vacuum_flags)) if cross_vacuum_flags else 0.0,
        "gcc_ratio": _summary_stats(gcc_ratios),
        "anisotropy": _summary_stats(anisotropies),
        "total_samples": int(z.shape[0]),
    }
    success_manifest: List[Dict[str, Any]] = []
    if success_top_k > 0 and per_sample:
        sorted_samples = sorted(
            per_sample,
            key=lambda row: row["formation_energy_per_atom"]
            if row.get("formation_energy_per_atom") is not None
            else float("inf"),
        )
        top_rows = sorted_samples[:success_top_k]
        for row in top_rows:
            success_manifest.append(
                {
                    "id": row["id"],
                    "formation_energy_per_atom": row.get("formation_energy_per_atom"),
                    "energy_mlip": row.get("energy_mlip"),
                    "success": row["success"],
                    "fail_reason": row["fail_reason"],
                }
            )
    return per_sample, tier0, tier1, success_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate token-based samples (Tier-0/1).")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.npz")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for eval artifacts.")
    parser.add_argument(
        "--self-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a minimal MIC sanity check and exit (does not require --samples).",
    )
    parser.add_argument("--stats-npz", type=Path, default=None, help="NPZ for volume bounds (p1/p99).")
    parser.add_argument(
        "--vacuum-min",
        type=float,
        default=15.0,
        help="If set, report vacuum_ok_rate = P(vacuum >= vacuum_min).",
    )
    parser.add_argument(
        "--element-refs",
        type=Path,
        default=Path("data/ref_energies.json"),
        help="JSON file mapping element symbols to reference energies (per atom). Set to empty string to disable.",
    )
    parser.add_argument(
        "--formation-energy-threshold",
        type=float,
        default=0.0,
        help="Threshold for formation energy per atom when calculating success rate.",
    )
    parser.add_argument(
        "--formation-energy-default-mu",
        type=float,
        default=0.0,
        help="Default elemental chemical potential used when a reference energy is missing.",
    )
    parser.add_argument(
        "--success-top-k",
        type=int,
        default=10,
        help="Number of top samples (sorted by formation energy) to record in success_manifest.",
    )
    parser.add_argument("--target-spacegroup", type=int, default=None)
    parser.add_argument("--spacegroup-symprec", type=float, default=1e-2)
    parser.add_argument(
        "--cond-max",
        type=float,
        default=None,
        help="If set, mark samples invalid when Gram condition exceeds this value.",
    )
    parser.set_defaults(
        min_dist=1.5,
        eval_min_dist=None,
        bond_cut=3.0,
        dup_eps=1e-3,
        v_min=None,
        v_max=None,
        sample=False,
        checkpoint=None,
        sample_out_dir=None,
        sample_args="",
        pbc_mask="1,1,0",
        self_check=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_min_dist is not None:
        print("[warn] --eval-min-dist is an alias for --min-dist; prefer --min-dist.")
        args.min_dist = float(args.eval_min_dist)
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    if args.self_check:
        lattice = np.eye(3, dtype=float)
        frac = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.9]], dtype=float)
        d_3d, _, shifts_3d = min_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 1))
        d_slab, _, shifts_slab = min_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 0))
        assert abs(d_3d - (0.2**2 + 0.0**2 + 0.2**2) ** 0.5) < 1e-6, d_3d
        assert abs(d_slab - (0.2**2 + 0.0**2 + 0.8**2) ** 0.5) < 1e-6, d_slab
        assert np.all(shifts_slab[..., 2] == 0.0)
        print("self-check passed")
        return

    if args.sample and args.samples is not None:
        raise ValueError("Use either --samples or --sample, not both.")
    if args.sample:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when using --sample.")
        from twodgen.scrip import sample_tokens as sample_tokens_mod

        sample_argv = ["--checkpoint", str(args.checkpoint)]
        if args.sample_out_dir is not None:
            sample_argv += ["--out-dir", str(args.sample_out_dir)]
        if args.sample_args:
            sample_argv += shlex.split(args.sample_args)
        sample_args = sample_tokens_mod.parse_args(sample_argv)
        samples_path = sample_tokens_mod.run_sampling(sample_args)
        samples = dict(np.load(samples_path))
        out_dir = args.out_dir or (samples_path.parent / "eval")
    else:
        if args.samples is None:
            raise ValueError("--samples is required unless --self-check is set.")
        samples_path = args.samples
        samples = dict(np.load(args.samples))
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = args.samples.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache = load_eval_cache(samples_path, pbc_mask=pbc_mask, bond_cut=args.bond_cut)
        for key in ("energy_mlip", "relaxed_flag", "cross_vacuum_flag"):
            if key not in samples and key in cache:
                samples[key] = cache[key]
    except Exception:
        pass

    if "min_dist_cut" in samples:
        stored = float(np.asarray(samples["min_dist_cut"]).reshape(-1)[0])
        if abs(stored - args.min_dist) > 1e-6:
            print(
                f"[warn] eval --min-dist ({args.min_dist}) differs from samples.npz min_dist_cut ({stored})."
            )

    v_min = args.v_min
    v_max = args.v_max
    if args.stats_npz is not None:
        stats = _load_npz_stats(args.stats_npz)
        if stats is not None:
            v_min, v_max = stats

    element_refs_path = args.element_refs
    if element_refs_path is not None and str(element_refs_path).strip() == "":
        element_refs_path = None
    element_refs = _load_element_refs(element_refs_path)
    atomic_ref_map = _atomic_ref_map(element_refs, float(args.formation_energy_default_mu))
    cond_max = args.cond_max
    if cond_max is None:
        if "project_gram_max_cond" in samples:
            try:
                cond_max = float(np.asarray(samples["project_gram_max_cond"]).reshape(-1)[0])
            except Exception:
                cond_max = None

    per_sample, tier0, tier1, success_manifest = _eval_samples(
        samples,
        v_min=v_min,
        v_max=v_max,
        min_dist_cut=args.min_dist,
        bond_cut=args.bond_cut,
        dup_eps=args.dup_eps,
        pbc_mask=pbc_mask,
        vacuum_min=args.vacuum_min,
        atomic_ref_map=atomic_ref_map,
        formation_energy_threshold=args.formation_energy_threshold,
        success_top_k=args.success_top_k,
        target_spacegroup=args.target_spacegroup,
        spacegroup_symprec=args.spacegroup_symprec,
        cond_max=cond_max,
    )
    eval_params = build_eval_params(
        min_dist_cut=float(args.min_dist),
        bond_cut=float(args.bond_cut),
        dup_eps=float(args.dup_eps),
        vacuum_min=args.vacuum_min,
        v_min=v_min,
        v_max=v_max,
        pbc_mask=pbc_mask,
        formation_energy_threshold=args.formation_energy_threshold,
        element_refs_path=element_refs_path,
        target_spacegroup=args.target_spacegroup,
        spacegroup_symprec=args.spacegroup_symprec,
        cond_max=cond_max,
    )
    write_eval_outputs(
        out_dir=out_dir,
        per_sample=per_sample,
        tier0=tier0,
        tier1=tier1,
        eval_params=eval_params,
        success_manifest=success_manifest,
        run_context={"source": "eval_samples", "samples": str(args.samples)},
    )

    print(f"Saved eval outputs to {out_dir}")


if __name__ == "__main__":
    main()
