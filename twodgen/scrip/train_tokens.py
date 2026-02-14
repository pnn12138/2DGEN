from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler

from twodgen.data.c2db_dataset import C2DBTokenNPZDataset
from twodgen.data.splits import load_c2db_split, select_split_indices, validate_split_indices
from twodgen.common.crystal import gram6_to_lattice, gram6_to_cholesky6, frac_mic_dist
from twodgen.common.run_metadata import collect_run_metadata
from twodgen.loss.schedule import LossWeightScheduleConfig, LossWeightScheduler
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.model.atom_transformer import AtomTransformerConfig
from twodgen.model.model_sizes import resolve_model_hparams


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        if isinstance(sample, dict):
            out = dict(sample)
            out["index"] = idx
            return out
        raise TypeError("Expected dataset to return a dict for training.")

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        keys = batch[0].keys()
        out = {}
        for key in keys:
            if key == "index":
                out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
            else:
                out[key] = torch.stack([b[key] for b in batch], dim=0)
        return out


def _unwrap_indexed_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, IndexedDataset):
        dataset = dataset.base
    return dataset


def _parse_loss_weight_keys(value: str) -> list[str]:
    return [key.strip().lower() for key in value.split(",") if key.strip()]


def _apply_loss_weights(model: AtomDenoiser, weights: dict[str, float]) -> None:
    cfg = model.loss_fn.cfg
    cfg.lambda_vacuum = float(weights.get("vacuum", cfg.lambda_vacuum))
    cfg.lambda_cross_vacuum = float(weights.get("cross_vacuum", cfg.lambda_cross_vacuum))
    cfg.lambda_cond = float(weights.get("cond", cfg.lambda_cond))
    cfg.lambda_chol_bound = float(weights.get("chol_bound", cfg.lambda_chol_bound))
    cfg.lambda_expand_collision = float(weights.get("expand_collision", cfg.lambda_expand_collision))
    cfg.lambda_volume = float(weights.get("volume", cfg.lambda_volume))
    cfg.lambda_c_len = float(weights.get("c_len", cfg.lambda_c_len))
    cfg.lambda_anisotropy = float(weights.get("anisotropy", cfg.lambda_anisotropy))


def _linear_warmup_factor(step: int, warmup_steps: int, start: float, end: float) -> float:
    if warmup_steps <= 0:
        return float(end)
    progress = float(step + 1) / float(max(1, warmup_steps))
    progress = max(0.0, min(progress, 1.0))
    return float(start + (end - start) * progress)


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.norm(2).item()) ** 2
    return total**0.5


def _grad_norm_by_prefix(model: nn.Module, prefixes: list[str]) -> dict[str, float]:
    norms: dict[str, float] = {}
    if not prefixes:
        return norms
    named_params = list(model.named_parameters())
    for prefix in prefixes:
        grads = [p.grad.detach() for name, p in named_params if name.startswith(prefix) and p.grad is not None]
        if not grads:
            norms[prefix] = float("nan")
            continue
        stacked = torch.stack([g.norm(2) for g in grads])
        norms[prefix] = float(torch.norm(stacked))
    return norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train token-based crystal diffusion model.")
    parser.add_argument("--npz", type=Path, default=None, help="Preprocessed token cache (npz).")
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional split json produced by twodgen.data.create_c2db_split (train/heldout indices).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "heldout"],
        help="Which split subset to use when --split-json is provided.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max training steps; if set, overrides epochs-based stopping.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Steps between stdout/jsonl logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (set 0 to avoid multiprocessing).",
    )
    parser.add_argument(
        "--tb-logdir",
        type=Path,
        default=None,
        help="TensorBoard log directory (disabled if unset).",
    )
    parser.add_argument(
        "--tb-interval",
        type=int,
        default=200,
        help="Steps between TensorBoard histogram logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Enable Weights & Biases logging to this project (optional).",
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (optional).")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional).")
    parser.add_argument(
        "--alert-steps",
        type=int,
        default=10000,
        help="Emit alerts only before this global step.",
    )
    parser.add_argument(
        "--alert-collision-rate",
        type=float,
        default=0.4,
        help="Alert if collision_rate exceeds this threshold.",
    )
    parser.add_argument(
        "--alert-min-dist-p10",
        type=float,
        default=0.8,
        help="Alert if min_dist_p10 falls below this threshold.",
    )
    parser.add_argument(
        "--alert-vacuum-gap",
        type=float,
        default=1.0,
        help="Alert if vacuum_gap_mean exceeds this threshold.",
    )
    parser.add_argument(
        "--alert-chol-clamp-rate",
        type=float,
        default=0.5,
        help="Alert if chol_log_clamp_rate exceeds this threshold.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl"],
        help="Model size preset.",
    )
    parser.add_argument(
        "--tail-adapter",
        type=str,
        default="none",
        choices=["none", "egnn"],
        help="Optional equivariant tail adapter (none/egnn).",
    )
    parser.add_argument("--tail-hidden-dim", type=int, default=128)
    parser.add_argument("--tail-scale", type=float, default=0.1)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--min-dist-train-cut",
        type=float,
        default=1.5,
        help="Distance cutoff (Angstrom) for collision penalty during training.",
    )
    parser.add_argument(
        "--min-dist-train-weight",
        type=float,
        default=0.12,
        help="Weight for collision penalty during training (0 disables).",
    )
    parser.add_argument(
        "--filter-min-dist-below",
        type=float,
        default=1.35,
        help="Optionally drop extreme-collision training samples with min_dist below this value.",
    )
    parser.add_argument(
        "--curriculum-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Curriculum learning: start with non-collision samples and gradually introduce collision-risk samples.",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=20,
        help="Number of epochs to linearly ramp collision-risk samples from 0%% to 100%%.",
    )
    parser.add_argument(
        "--curriculum-min-dist-cut",
        type=float,
        default=1.5,
        help="Threshold (Angstrom) to classify collision-risk samples for curriculum.",
    )
    parser.add_argument(
        "--quality-jsonl",
        type=Path,
        default=Path("data/C2DB/c2db_quality.jsonl"),
        help="Path to `c2db_quality.jsonl` produced by `clean_c2db_2d.py` for filtering training rows.",
    )
    parser.add_argument(
        "--quality-buckets",
        type=str,
        default="good,risk",
        help="Comma-separated list of quality buckets to keep when `--quality-jsonl` is provided.",
    )
    parser.add_argument(
        "--quality-hard-pass-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using `--quality-jsonl`, drop rows flagged as hard fails.",
    )
    parser.add_argument(
        "--comp-loss-weight",
        type=float,
        default=1.0,
        help="Weight for composition consistency loss (counts_vector vs predicted counts).",
    )
    parser.add_argument(
        "--comp-loss-mode",
        type=str,
        default="l1",
        choices=["l1", "cosine"],
        help="Composition loss mode.",
    )
    parser.add_argument(
        "--vacuum-loss-weight",
        type=float,
        default=1.0,
        help="Weight for 2D vacuum loss (0 disables).",
    )
    parser.add_argument(
        "--vacuum-min",
        type=float,
        default=15.0,
        help="Minimum vacuum thickness (Angstrom) for vacuum loss.",
    )
    parser.add_argument(
        "--vacuum-loss-mode",
        type=str,
        default="c_axis",
        choices=["vacuum_gap", "c_axis"],
        help="Vacuum loss mode: vacuum_gap uses max gap; c_axis uses hinge(c_len).",
    )
    parser.add_argument(
        "--vacuum-loss-power",
        type=int,
        default=2,
        help="Power for vacuum loss: 1=linear, 2=squared, ...",
    )
    parser.add_argument(
        "--cross-vacuum-loss-weight",
        type=float,
        default=0.0,
        help="Weight for cross-vacuum proxy loss during training (0 disables).",
    )
    parser.add_argument(
        "--cross-vacuum-bond-cut",
        type=float,
        default=3.0,
        help="Bond cutoff (Angstrom) for cross-vacuum proxy.",
    )
    parser.add_argument(
        "--cross-vacuum-power",
        type=int,
        default=2,
        help="Power for cross-vacuum proxy loss: 1=linear, 2=squared, ...",
    )
    parser.add_argument(
        "--symmetry-loss-weight",
        type=float,
        default=0.0,
        help="Weight for spacegroup mismatch penalty (requires spacegroup_number in dataset).",
    )
    parser.add_argument(
        "--symmetry-mode",
        type=str,
        default="off",
        choices=["off", "soft", "hard"],
        help="Symmetry control mode shared by train/sample: off | soft | hard.",
    )
    parser.add_argument("--symmetry-symprec", type=float, default=1e-2)
    parser.add_argument(
        "--wyckoff-constraint",
        type=str,
        default="off",
        choices=["off", "soft", "hard"],
        help="Wyckoff-level constraint placeholder (metadata only in current implementation).",
    )
    parser.add_argument(
        "--angle-loss-weight",
        type=float,
        default=0.1,
        help="Weight for angle constraint loss during training.",
    )
    parser.add_argument(
        "--angle-min",
        type=float,
        default=30.0,
        help="Minimum allowed lattice angle (degrees) for angle loss.",
    )
    parser.add_argument(
        "--angle-max",
        type=float,
        default=150.0,
        help="Maximum allowed lattice angle (degrees) for angle loss.",
    )
    parser.add_argument(
        "--angle-param-mode",
        type=str,
        default="raw",
        choices=["raw", "cos", "sigmoid"],
        help="Angle parameterization mode for angle constraint (raw/cos/sigmoid).",
    )
    parser.add_argument(
        "--angle-sigmoid-tau",
        type=float,
        default=10.0,
        help="Temperature for sigmoid angle parameterization (only when angle-param-mode=sigmoid).",
    )
    parser.add_argument(
        "--cond-loss-weight",
        type=float,
        default=0.01,
        help="Weight for Gram condition penalty during training.",
    )
    parser.add_argument(
        "--cond-max",
        type=float,
        default=1e3,
        help="Maximum allowed Gram condition number for cond loss.",
    )
    parser.add_argument(
        "--cond-max-start",
        type=float,
        default=None,
        help="Optional start value for cond_max schedule; if set, cond_max will be scheduled each step.",
    )
    parser.add_argument(
        "--cond-max-end",
        type=float,
        default=None,
        help="Optional end value for cond_max schedule; defaults to --cond-max when start is set.",
    )
    parser.add_argument(
        "--cond-max-steps",
        type=int,
        default=None,
        help="Number of steps for cond_max schedule (after此保持 end 值)。",
    )
    parser.add_argument(
        "--cond-max-schedule",
        type=str,
        choices=["linear", "cosine"],
        default="linear",
        help="Schedule shape for cond_max when start/end/steps are provided.",
    )
    parser.add_argument(
        "--c-len-loss-weight",
        type=float,
        default=0.05,
        help="Weight for c-axis length penalty during training.",
    )
    parser.add_argument(
        "--c-len-min",
        type=float,
        default=15.0,
        help="Minimum c-axis length enforced by the c_len penalty.",
    )
    parser.add_argument(
        "--volume-loss-weight",
        type=float,
        default=0.1,
        help="Weight for lattice volume penalty during training.",
    )
    parser.add_argument(
        "--volume-min",
        type=float,
        default=131.0,
        help="Minimum lattice volume enforced by the volume penalty.",
    )
    parser.add_argument(
        "--volume-max",
        type=float,
        default=1900.0,
        help="Optional maximum lattice volume enforced by the volume penalty.",
    )
    parser.add_argument(
        "--auto-volume-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-set volume_min/volume_max from train split p1/p99 statistics.",
    )
    parser.add_argument(
        "--anisotropy-loss-weight",
        type=float,
        default=0.0,
        help="Weight for anisotropy collapse penalty (0 disables).",
    )
    parser.add_argument(
        "--anisotropy-min-std",
        type=float,
        default=1.0,
        help="Minimum std of lattice lengths used by anisotropy loss.",
    )
    parser.add_argument(
        "--loss-hinge",
        type=str,
        choices=["relu", "softplus"],
        default="relu",
        help="Hinge shape for volume/c_len/anisotropy losses.",
    )
    parser.add_argument(
        "--loss-softplus-beta",
        type=float,
        default=10.0,
        help="Softplus beta when --loss-hinge=softplus.",
    )
    parser.add_argument(
        "--chol-bound-loss-weight",
        type=float,
        default=0.05,
        help="Weight for Cholesky boundary penalty during training (0 disables).",
    )
    parser.add_argument(
        "--chol-bound-margin",
        type=float,
        default=0.2,
        help="Soft margin (log space) before chol bound penalty activates.",
    )
    parser.add_argument(
        "--chol-bound-power",
        type=int,
        default=2,
        help="Power for chol bound penalty.",
    )
    parser.add_argument(
        "--expand-on-collision-weight",
        type=float,
        default=0.05,
        help="Weight for expand-on-collision loss using predicted x0 (0 disables).",
    )
    parser.add_argument(
        "--expand-on-collision-cut",
        type=float,
        default=1.5,
        help="Min distance cutoff (Angstrom) for expand-on-collision loss.",
    )
    parser.add_argument(
        "--loss-weight-warmup-steps",
        type=int,
        default=15000,
        help="Warmup steps for selected loss weights (0 disables).",
    )
    parser.add_argument(
        "--loss-weight-warmup-keys",
        type=str,
        default="vacuum,cross_vacuum,cond,chol_bound,expand_collision",
        help="Comma-separated loss keys for warmup (vacuum,cross_vacuum,cond,chol_bound,expand_collision).",
    )
    parser.add_argument(
        "--loss-weight-warmup-start",
        type=float,
        default=0.0,
        help="Start factor for loss weight warmup.",
    )
    parser.add_argument(
        "--loss-weight-warmup-end",
        type=float,
        default=1.0,
        help="End factor for loss weight warmup.",
    )
    parser.add_argument(
        "--loss-weight-schedule",
        type=str,
        choices=["linear", "sigmoid", "cosine"],
        default="sigmoid",
        help="Schedule shape for warmup (linear/sigmoid/cosine).",
    )
    parser.add_argument(
        "--volume-c-len-warmup-steps",
        type=int,
        default=2000,
        help="Linear warmup steps for volume/c_len weights (0 disables).",
    )
    parser.add_argument(
        "--volume-c-len-warmup-start",
        type=float,
        default=0.0,
        help="Start factor for volume/c_len linear warmup.",
    )
    parser.add_argument(
        "--volume-c-len-warmup-end",
        type=float,
        default=1.0,
        help="End factor for volume/c_len linear warmup.",
    )
    parser.add_argument(
        "--debug-grad-submodules",
        type=str,
        default="",
        help="Comma-separated list of parameter name prefixes; log their grad_norm each log_interval.",
    )
    parser.add_argument(
        "--chol-log-relax",
        type=float,
        default=0.15,
        help="Relaxation margin added to chol_log_min/max during projection and sampling.",
    )
    parser.add_argument(
        "--noise-scale-zn",
        type=float,
        default=0.3,
        help="Noise scale for z_norm diffusion target.",
    )

    parser.set_defaults(
        deterministic=False,
        embed_dim=None,
        depth=None,
        num_heads=None,
        mlp_ratio=None,
        dropout=None,
        time_embed_dim=None,
        z_embed_dim=None,
        f_embed_dim=None,
        rbf_dim=None,
        pair_mlp_hidden=None,
        optimizer="adamw",
        weight_decay=1e-2,
        betas="0.9,0.95",
        warmup_steps=500,
        min_lr=1e-6,
        lr_schedule="cosine",
        clip_grad=1.0,
        ema=True,
        ema_decay=0.9999,
        num_workers=4,
        log_interval=50,
        max_atoms=24,
        g_scale=100.0,
        k_neighbors=32,
        dual_graph=False,
        edge_type_dim=0,
        edge_type_gating=True,
        wrap_embed_dim=0,
        max_steps=None,
        mode="diffusion",
        no_uncertainty_weighting=False,
        drop_last=True,
        cell_rep="cholesky6",
        chol_log_min=None,
        chol_log_max=None,
        cell_init="iso",
        cell_init_scale=None,
        cell_init_noise=None,
        cell_init_scale_factor=1.5,
        cell_log_min_factor=0.7,
        cell_log_max_factor=2.5,
        use_geometry_fields=True,
        align_atoms=True,
        coord_frame="canon",
        bucket_batches=False,
        bucket_shuffle=False,
        use_condition=True,
        cond_drop_prob=0.1,
        cond_fields="counts_vector,lattice_param,t",
        cond_normalize_fields="lattice_param,t",
        use_comp_encoder=True,
        comp_embed_dim=64,
        comp_pool_mode="count",
        comp_use_frac=True,
        element_ids=None,
        pbc_mask="1,1,0",
    )
    return parser.parse_args()


def _parse_pbc_mask(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def _parse_cond_fields(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_element_ids(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        return None
    if any(elem <= 0 for elem in ids):
        raise ValueError("--element-ids must be positive integers (Z).")
    return ids


_VALID_QUALITY_BUCKETS = {"good", "risk", "bad"}
_DEFAULT_QUALITY_BUCKETS = ("good", "risk")
_SPACEGROUP_MAX = 230
_SPACEGROUP_FIELDS = {"spacegroup", "spacegroup_number", "spacegroup_num"}


def _parse_quality_buckets(value: Optional[str]) -> list[str]:
    if value is None:
        return list(_DEFAULT_QUALITY_BUCKETS)
    buckets = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not buckets:
        return list(_DEFAULT_QUALITY_BUCKETS)
    invalid = [bucket for bucket in buckets if bucket not in _VALID_QUALITY_BUCKETS]
    if invalid:
        raise ValueError(
            "--quality-buckets contains unknown buckets "
            f"{invalid!r} (supported: {_VALID_QUALITY_BUCKETS})"
        )
    # Preserve order while deduplicating
    seen: set[str] = set()
    ordered: list[str] = []
    for bucket in buckets:
        if bucket in seen:
            continue
        seen.add(bucket)
        ordered.append(bucket)
    return ordered


def _load_quality_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            material_id = str(row.get("material_id", "")).strip()
            if not material_id:
                continue
            quality_bucket = str(row.get("quality_bucket", "")).lower()
            mapping[material_id] = {
                "quality_bucket": quality_bucket,
                "hard_pass": bool(row.get("hard_pass")),
            }
    return mapping


def _material_id_for(dataset: Dataset, idx: int) -> str | None:
    dataset_for_iter = _unwrap_indexed_dataset(dataset)
    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
        if dataset_for_iter.material_ids is None:
            return None
        return dataset_for_iter.material_ids[idx]
    return None


def _filter_indices_by_quality(
    dataset: Dataset,
    indices: list[int],
    quality_map: dict[str, dict[str, Any]],
    allowed_buckets: list[str],
    require_hard_pass: bool,
) -> tuple[list[int], int, int]:
    allowed: set[str] | None = set(allowed_buckets) if allowed_buckets else None
    kept: list[int] = []
    missing = 0
    filtered = 0
    for idx in indices:
        material_id = _material_id_for(dataset, idx)
        if material_id is None:
            missing += 1
            continue
        entry = quality_map.get(material_id)
        if entry is None:
            missing += 1
            continue
        bucket = entry.get("quality_bucket") or ""
        if allowed is not None and bucket not in allowed:
            filtered += 1
            continue
        if require_hard_pass and not entry.get("hard_pass", False):
            filtered += 1
            continue
        kept.append(idx)
    return kept, missing, filtered


def _parse_betas(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--betas must have two comma-separated values, e.g. 0.9,0.95")
    return float(parts[0]), float(parts[1])


def _resolve_cond_fields(args: argparse.Namespace) -> list[str]:
    fields = _parse_cond_fields(args.cond_fields)
    if fields:
        return fields
    return ["counts_vector"]


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        counts: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        order = torch.argsort(counts)
        self.buckets = [
            order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)
        ]
        if self.shuffle:
            perm = torch.randperm(len(self.buckets), generator=self.generator).tolist()
            self.buckets = [self.buckets[i] for i in perm]
            for bucket in self.buckets:
                perm_in = torch.randperm(len(bucket), generator=self.generator).tolist()
                reordered = [bucket[i] for i in perm_in]
                bucket[:] = reordered

    def __iter__(self):
        yield from self.buckets

    def __len__(self) -> int:
        return len(self.buckets)


def _atom_counts(dataset: C2DBTokenNPZDataset) -> torch.Tensor:
    dataset_for_iter = _unwrap_indexed_dataset(dataset)
    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
        return dataset_for_iter.atom_mask.sum(dim=1)
    counts = []
    for i in range(len(dataset_for_iter)):
        counts.append(dataset_for_iter[i]["atom_mask"].sum())
    return torch.stack(counts, dim=0)


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def prepare_dataloader(
    dataset: C2DBTokenNPZDataset,
    batch_size: int,
    num_workers: int,
    use_buckets: bool,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    indices: list[int] | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id + 1
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    from torch.utils.data import Sampler, SubsetRandomSampler

    class _SubsetSequentialSampler(Sampler[int]):
        def __init__(self, subset: list[int]) -> None:
            self.subset = subset

        def __iter__(self):
            yield from self.subset

        def __len__(self) -> int:
            return len(self.subset)

    class _IndexBucketBatchSampler(Sampler[list[int]]):
        def __init__(self, subset: list[int], counts_subset: torch.Tensor) -> None:
            order_local = torch.argsort(counts_subset).tolist()
            mapped = [subset[i] for i in order_local]
            self.buckets = [
                mapped[i : i + batch_size] for i in range(0, len(mapped), batch_size)
            ]
            if shuffle:
                perm = torch.randperm(len(self.buckets), generator=generator).tolist()
                self.buckets = [self.buckets[i] for i in perm]
                for bucket in self.buckets:
                    perm_in = torch.randperm(len(bucket), generator=generator).tolist()
                    bucket[:] = [bucket[i] for i in perm_in]

        def __iter__(self):
            yield from self.buckets

        def __len__(self) -> int:
            return len(self.buckets)

    if indices is not None:
        validate_split_indices(indices, total=len(dataset))

    if use_buckets:
        counts_full = _atom_counts(dataset).float()
        if indices is None:
            batch_sampler: Sampler[list[int]] = BucketBatchSampler(
                counts_full, batch_size=batch_size, shuffle=shuffle, generator=generator
            )
        else:
            batch_sampler = _IndexBucketBatchSampler(indices, counts_full[indices])
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )

    if indices is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=dataset.collate_fn,
            generator=generator,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )

    sampler: Sampler[int]
    if shuffle:
        sampler = SubsetRandomSampler(indices, generator=generator)
    else:
        sampler = _SubsetSequentialSampler(indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

@torch.no_grad()
def _compute_dataset_min_dist(
    dataset: C2DBTokenNPZDataset,
    *,
    pbc_mask: tuple[int, int, int],
    g_scale: float,
    batch_size: int = 256,
) -> torch.Tensor:
    dataset_for_iter = _unwrap_indexed_dataset(dataset)

    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
        if dataset_for_iter.min_dist is not None:
            return dataset_for_iter.min_dist.float().reshape(-1).cpu()
        cached = dataset_for_iter.extra.get("min_dist")
        if cached is not None:
            return cached.float().reshape(-1).cpu()
        use_canon = getattr(dataset_for_iter, "coord_frame_actual", "raw") == "canon"
        frac = dataset_for_iter.f_canon if use_canon and dataset_for_iter.f_canon is not None else dataset_for_iter.f
        mask = (
            dataset_for_iter.atom_mask_canon
            if use_canon and dataset_for_iter.atom_mask_canon is not None
            else dataset_for_iter.atom_mask
        )
        lattice = None
        if use_canon and dataset_for_iter.lattice_canon is not None:
            lattice = dataset_for_iter.lattice_canon
        elif dataset_for_iter.lattice is not None:
            lattice = dataset_for_iter.lattice
        if lattice is None:
            gram6 = dataset_for_iter.gram6_canon if use_canon and dataset_for_iter.gram6_canon is not None else dataset_for_iter.gram6
            lattice = gram6_to_lattice(gram6 * g_scale)

        n = frac.shape[0]
        out = torch.empty((n,), dtype=torch.float32)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            dist = frac_mic_dist(
                frac[start:end],
                lattice[start:end],
                mask[start:end],
                pbc_mask=pbc_mask,
            )
            out[start:end] = dist.amin(dim=(1, 2)).float().cpu()
        return out

    loader = DataLoader(
        dataset_for_iter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda items: {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]},
    )
    mins: list[torch.Tensor] = []
    for batch in loader:
        frac = batch["frac_coords"].float()
        atom_mask = batch["atom_mask"].float()
        gram6 = batch["gram6"].float()
        lattice = gram6_to_lattice(gram6 * g_scale)
        dist = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=pbc_mask)
        mins.append(dist.amin(dim=(1, 2)).float().cpu())
    return torch.cat(mins, dim=0) if mins else torch.empty((0,), dtype=torch.float32)


def _curriculum_indices(
    *,
    base_indices: list[int],
    min_dist_all: torch.Tensor,
    min_dist_cut: float,
    epoch: int,
    epochs_ramp: int,
    seed: int,
) -> list[int]:
    easy = [i for i in base_indices if float(min_dist_all[i]) >= min_dist_cut]
    hard = [i for i in base_indices if float(min_dist_all[i]) < min_dist_cut]
    if not hard or epochs_ramp <= 0:
        return base_indices
    frac = min(1.0, float(epoch) / float(epochs_ramp))
    k = int(round(len(hard) * frac))
    if k <= 0:
        return easy
    if k >= len(hard):
        return base_indices
    g = torch.Generator()
    g.manual_seed(seed + 1009 * epoch)
    perm = torch.randperm(len(hard), generator=g).tolist()
    selected = [hard[i] for i in perm[:k]]
    return easy + selected


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _compute_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if schedule == "constant" or total_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _adjust_lr(
    optimizer: optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    lr = _compute_lr(step, total_steps, warmup_steps, base_lr, min_lr, schedule)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _schedule_value(step: int, total_steps: int, start: float, end: float, schedule: str) -> float:
    if total_steps <= 0:
        return end
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    if schedule == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * progress


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}


def _gram6_to_scube(gram6: torch.Tensor, g_scale: float) -> torch.Tensor:
    g11, g22, g33, g12, g13, g23 = [gram6[:, i] for i in range(6)]
    G = torch.stack(
        [
            torch.stack([g11, g12, g13], dim=-1),
            torch.stack([g12, g22, g23], dim=-1),
            torch.stack([g13, g23, g33], dim=-1),
        ],
        dim=-2,
    )
    det_g = torch.linalg.det(G).clamp_min(1e-12)
    # NOTE:
    # - Dataset stores `gram6 = lattice_to_gram6(lattice) / g_scale`, so the Gram matrix here is
    #   `G_scaled = G_phys / g_scale`.
    # - When `cell_rep == cholesky6`, the model operates on the Cholesky parameters of `G_scaled`,
    #   and the physical lattice is recovered later by multiplying by `sqrt(g_scale)`.
    # Therefore, statistics used to set `chol_log_min/max` and `cell_init_scale` must be computed
    # in this *internal* (scaled) length unit, i.e. without multiplying by `sqrt(g_scale)`.
    return det_g.pow(1.0 / 6.0)


def _estimate_chol_diag_stats_from_lattice(
    lattice: torch.Tensor,
    g_scale: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None:
    """
    Estimate per-dimension Cholesky-diagonal log statistics in the *internal* (scaled) length unit.

    For cholesky6, the model operates on the Cholesky parameters of G_scaled = G_phys / g_scale, i.e.
    lattice_scaled = lattice_phys / sqrt(g_scale). The returned diag logs are for that scaled lattice.
    """
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattice shape (B,3,3), got {tuple(lattice.shape)}")
    if lattice.numel() == 0:
        return None
    lattice = lattice.float()
    scale = math.sqrt(float(g_scale)) if float(g_scale) > 0 else 1.0
    lattice_scaled = lattice / max(scale, 1e-12)
    gram = lattice_scaled @ lattice_scaled.transpose(-1, -2)
    gram = 0.5 * (gram + gram.transpose(-1, -2))

    eye = torch.eye(3, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    jitter = 1e-6
    gram_work = gram
    L, info = torch.linalg.cholesky_ex(gram_work + jitter * eye)
    tries = 0
    while info.any() and tries < 4:
        jitter *= 10.0
        L, info = torch.linalg.cholesky_ex(gram_work + jitter * eye)
        tries += 1
    if info.any():
        bad = info > 0
        vals, vecs = torch.linalg.eigh(gram_work[bad])
        vals = vals.clamp_min(1e-6)
        gram_fixed = vecs @ torch.diag_embed(vals) @ vecs.transpose(-1, -2)
        L_bad, info_bad = torch.linalg.cholesky_ex(gram_fixed + 1e-6 * eye)
        if info_bad.any():
            return None
        L[bad] = L_bad

    diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))  # (B,3)
    diag = diag[torch.isfinite(diag).all(dim=-1)]
    if diag.numel() == 0:
        return None
    q = torch.quantile(diag, torch.tensor([0.1, 0.5, 0.9], device=diag.device, dtype=diag.dtype), dim=0)
    q10, q50, q90 = q[0], q[1], q[2]
    log_std = float(torch.std(diag.reshape(-1), unbiased=False).clamp_min(1e-6).item())
    return (
        (float(q10[0]), float(q10[1]), float(q10[2])),
        (float(q50[0]), float(q50[1]), float(q50[2])),
        (float(q90[0]), float(q90[1]), float(q90[2])),
        log_std,
    )


def _estimate_scube_stats(
    dataset: C2DBTokenNPZDataset, g_scale: float, indices: list[int] | None = None
) -> tuple[float, float, float, float]:
    if isinstance(dataset, C2DBTokenNPZDataset):
        gram6 = dataset.gram6.float()
        if indices is not None:
            gram6 = gram6[indices]
    else:
        gram6_list = []
        if indices is None:
            indices = list(range(len(dataset)))
        for i in indices:
            gram6_list.append(dataset[i]["gram6"].float())
        gram6 = torch.stack(gram6_list, dim=0)
    scube = _gram6_to_scube(gram6, g_scale).cpu().numpy()
    scube = scube[np.isfinite(scube)]
    if scube.size == 0:
        return 1.0, 1.0, 1.0, 0.1
    s10 = float(np.percentile(scube, 10.0))
    s50 = float(np.percentile(scube, 50.0))
    s90 = float(np.percentile(scube, 90.0))
    log_std = float(np.std(np.log(scube + 1e-12)))
    return s10, s50, s90, log_std


def _estimate_volume_stats(
    dataset: C2DBTokenNPZDataset,
    g_scale: float,
    indices: list[int] | None = None,
) -> tuple[float, float, float] | None:
    if isinstance(dataset, C2DBTokenNPZDataset) and dataset.lattice is not None:
        lattice = dataset.lattice.float()
        if indices is not None:
            lattice = lattice[indices]
        vols = torch.abs(torch.linalg.det(lattice))
    else:
        if isinstance(dataset, C2DBTokenNPZDataset):
            gram6 = dataset.gram6.float()
            if indices is not None:
                gram6 = gram6[indices]
        else:
            gram6_list = []
            if indices is None:
                indices = list(range(len(dataset)))
            for i in indices:
                gram6_list.append(dataset[i]["gram6"].float())
            gram6 = torch.stack(gram6_list, dim=0)
        lattice = gram6_to_lattice(gram6 * float(g_scale))
        vols = torch.abs(torch.linalg.det(lattice))
    vols = vols[torch.isfinite(vols)]
    if vols.numel() == 0:
        return None
    q = torch.quantile(vols, torch.tensor([0.01, 0.5, 0.99], device=vols.device))
    return float(q[0]), float(q[1]), float(q[2])


def _counts_from_z(z: torch.Tensor, atom_mask: torch.Tensor, num_elements: int) -> torch.Tensor:
    bsz, n = z.shape
    counts = torch.zeros(bsz, num_elements, device=z.device, dtype=torch.float32)
    valid = atom_mask > 0.5
    z_valid = z[valid].long()
    batch_idx = torch.arange(bsz, device=z.device).unsqueeze(1).expand_as(valid)[valid]
    elem_idx = z_valid - 1
    keep = (elem_idx >= 0) & (elem_idx < num_elements)
    if keep.any():
        counts.index_put_(
            (batch_idx[keep], elem_idx[keep]),
            torch.ones_like(elem_idx[keep], dtype=counts.dtype),
            accumulate=True,
        )
    return counts


def _build_condition(
    batch: dict,
    max_atoms: int,
    num_elements: int,
    cond_stats: dict | None = None,
    cond_fields: Optional[list[str]] = None,
) -> torch.Tensor:
    fields = cond_fields or []
    parts = []
    for field in fields:
        if field in ("counts", "counts_vector"):
            if "counts_vector" in batch:
                counts = batch["counts_vector"].float()
            else:
                counts = _counts_from_z(batch["atomic_numbers"], batch["atom_mask"], num_elements)
            counts = counts / max(1, max_atoms)
            parts.append(counts)
            continue
        if field in _SPACEGROUP_FIELDS:
            key = "spacegroup_number" if "spacegroup_number" in batch else field
            if key not in batch:
                raise ValueError(f"{field} not found in batch but requested by --cond-fields.")
            sg = batch[key].long().view(-1)
            one_hot = torch.zeros(sg.shape[0], _SPACEGROUP_MAX, device=sg.device, dtype=torch.float32)
            valid = (sg > 0) & (sg <= _SPACEGROUP_MAX)
            if valid.any():
                idx = sg[valid] - 1
                one_hot[valid, idx] = 1.0
            parts.append(one_hot)
            continue
        if field not in batch:
            raise ValueError(f"{field} not found in batch but requested by --cond-fields.")
        value = batch[field].float()
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if cond_stats is not None and f"{field}_mean" in cond_stats and f"{field}_std" in cond_stats:
            mean = cond_stats[f"{field}_mean"].to(value.device, value.dtype)
            std = cond_stats[f"{field}_std"].to(value.device, value.dtype)
            value = (value - mean) / std
        parts.append(value)
    if not parts:
        raise ValueError("Condition fields resolved to empty list.")
    return torch.cat(parts, dim=-1)


def _infer_cond_dim(sample: dict, cond_fields: list[str], num_elements: int) -> int:
    cond_dim = 0
    for field in cond_fields:
        if field in ("counts", "counts_vector"):
            cond_dim += num_elements
            continue
        if field in _SPACEGROUP_FIELDS:
            key = "spacegroup_number" if "spacegroup_number" in sample else field
            if key not in sample:
                raise ValueError(f"{field} not found but requested by --cond-fields.")
            cond_dim += _SPACEGROUP_MAX
            continue
        if field not in sample:
            raise ValueError(f"{field} not found but requested by --cond-fields.")
        value = sample[field]
        cond_dim += int(value.numel())
    return cond_dim


def _compute_cond_stats(
    dataset: C2DBTokenNPZDataset,
    cond_fields: list[str],
    normalize_fields: list[str],
    indices: list[int] | None = None,
    t_stats: tuple[float, float] | None = None,
) -> dict:
    stats: dict[str, torch.Tensor] = {}
    if not isinstance(dataset, C2DBTokenNPZDataset):
        return stats
    indices_t = torch.as_tensor(indices, dtype=torch.long) if indices is not None else None
    for field in normalize_fields:
        if field in ("counts", "counts_vector"):
            continue
        if field in _SPACEGROUP_FIELDS:
            continue
        if field not in cond_fields:
            continue
        if field == "t" and t_stats is not None:
            stats["t_mean"] = torch.tensor(t_stats[0])
            stats["t_std"] = torch.tensor(t_stats[1])
            continue
        value = getattr(dataset, field, None)
        if value is None:
            continue
        value = value.float()
        if indices_t is not None:
            value = value.index_select(0, indices_t)
        mean = value.mean(dim=0)
        std = value.std(dim=0, unbiased=False).clamp_min(1e-6)
        stats[f"{field}_mean"] = mean
        stats[f"{field}_std"] = std
    return stats


def train_one_epoch(
    model: AtomDenoiser,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    tb_writer: Any | None,
    tb_interval: int,
    wandb_run: Any | None,
    alert_steps: int,
    alert_collision_rate: float,
    alert_min_dist_p10: float,
    alert_vacuum_gap: float,
    alert_chol_clamp_rate: float,
    global_step: int,
    max_steps: int | None,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
    clip_grad: float,
    ema: EMA | None,
    use_condition: bool,
    use_geometry_fields: bool,
    use_thickness: bool,
    max_atoms: int,
    num_elements: int,
    cond_stats: dict | None = None,
    cond_fields: Optional[list[str]] = None,
    t_stats: tuple[float, float] | None = None,
    metrics_log_path: Optional[Path] = None,
    loss_weight_scheduler: LossWeightScheduler | None = None,
    base_loss_weights: dict[str, float] | None = None,
    volume_c_len_warmup_steps: int = 0,
    volume_c_len_warmup_start: float = 0.0,
    volume_c_len_warmup_end: float = 1.0,
    cond_max_schedule: dict | None = None,
    debug_grad_prefixes: list[str] | None = None,
) -> tuple[int, float]:
    model.train()
    total_loss = 0.0
    step_count = 0
    debug_grad_prefixes = debug_grad_prefixes or []
    grad_prefix_norms: dict[str, float] | None = None
    for batch in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        z = batch["atomic_numbers"].to(device, non_blocking=True)
        frac = batch["frac_coords"].to(device, non_blocking=True)
        atom_mask = batch["atom_mask"].to(device, non_blocking=True)
        gram6 = batch["gram6"].to(device, non_blocking=True)
        batch_indices = batch.get("index")
        uv_angle = None
        z_norm = None
        lattice_param = None
        slab_t = None
        if use_geometry_fields:
            uv_angle = batch.get("uv_angle")
            z_norm = batch.get("z_norm")
            lattice_param = batch.get("lattice_param")
            if uv_angle is not None:
                uv_angle = uv_angle.to(device, non_blocking=True)
            if z_norm is not None:
                z_norm = z_norm.to(device, non_blocking=True)
            if lattice_param is not None:
                lattice_param = lattice_param.to(device, non_blocking=True)
        if use_thickness:
            slab_t = batch.get("t")
            if slab_t is not None:
                slab_t = slab_t.to(device, non_blocking=True)
                if t_stats is not None:
                    t_mean, t_std = t_stats
                    slab_t = (slab_t - t_mean) / t_std
        cond = None
        if use_condition:
            cond = _build_condition(
                batch,
                max_atoms=max_atoms,
                num_elements=num_elements,
                cond_stats=cond_stats,
                cond_fields=cond_fields,
            ).to(device, non_blocking=True)
        counts_vector = None
        if use_condition:
            counts_vector = batch.get("counts_vector")
            if counts_vector is not None:
                counts_vector = counts_vector.to(device, non_blocking=True)
            else:
                counts_vector = _counts_from_z(z, atom_mask, num_elements).to(device)
        spacegroup_number = None
        if "spacegroup_number" in batch:
            spacegroup_number = batch["spacegroup_number"].to(device, non_blocking=True)

        lr = _adjust_lr(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr, schedule)
        loss_weight_state = None
        if loss_weight_scheduler is not None:
            loss_weight_state = loss_weight_scheduler.weights(global_step)
            _apply_loss_weights(model, loss_weight_state)
        if (
            base_loss_weights is not None
            and volume_c_len_warmup_steps > 0
            and ("volume" in base_loss_weights or "c_len" in base_loss_weights)
        ):
            factor = _linear_warmup_factor(
                global_step,
                volume_c_len_warmup_steps,
                volume_c_len_warmup_start,
                volume_c_len_warmup_end,
            )
            if loss_weight_state is None:
                loss_weight_state = {}
            if "volume" in base_loss_weights:
                loss_weight_state["volume"] = float(base_loss_weights["volume"]) * factor
            if "c_len" in base_loss_weights:
                loss_weight_state["c_len"] = float(base_loss_weights["c_len"]) * factor
            _apply_loss_weights(model, loss_weight_state)
        optimizer.zero_grad(set_to_none=True)
        # Update cond_max schedule if requested
        if cond_max_schedule is not None:
            start = cond_max_schedule["start"]
            end = cond_max_schedule["end"]
            steps_sched = cond_max_schedule["steps"]
            mode = cond_max_schedule["schedule"]
            current = _schedule_value(global_step, steps_sched, start, end, mode)
            model.loss_fn.cfg.cond_max = float(current)
            if hasattr(model, "cfg") and hasattr(model.cfg, "diffusion"):
                model.cfg.diffusion.cond_max = float(current)
        loss, _, _, _, metrics = model(
            z,
            frac,
            atom_mask,
            gram6,
            cond,
            counts_vector,
            uv_angle=uv_angle,
            z_norm=z_norm,
            lattice_param=lattice_param,
            slab_t=slab_t,
            spacegroup_number=spacegroup_number,
        )
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        grad_norm = None
        if (
            global_step % log_interval == 0
            or (tb_writer is not None and global_step % tb_interval == 0)
            or wandb_run is not None
        ):
            grad_norm = _grad_norm(model.parameters())
            grad_prefix_norms = _grad_norm_by_prefix(model, debug_grad_prefixes)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        if global_step % log_interval == 0:
            with torch.no_grad():
                lattice = gram6_to_lattice(gram6 * model.cfg.model.g_scale)
                dist = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=model.cfg.model.pbc_mask)
                min_dist_batch = dist.amin(dim=(1, 2)).detach().cpu().numpy()
                atom_counts = atom_mask.sum(dim=1).detach().cpu().numpy()
                valid_mask = (atom_counts >= 2) & np.isfinite(min_dist_batch)
                min_dist_mean = (
                    float(np.mean(min_dist_batch[valid_mask])) if valid_mask.any() else float("nan")
                )
                min_dist_p10 = (
                    float(np.percentile(min_dist_batch[valid_mask], 10.0)) if valid_mask.any() else float("nan")
                )
                collision_rate = (
                    float(np.mean(min_dist_batch[valid_mask] < model.cfg.min_dist_train_cut))
                    if valid_mask.any()
                    else 0.0
                )
                min_dist_inf = int(np.sum(~np.isfinite(min_dist_batch)))
                min_dist_low_atoms = int(np.sum(atom_counts < 2))
                min_dist_inf_indices = None
                min_dist_low_atoms_indices = None
                if batch_indices is not None:
                    idx_np = batch_indices.detach().cpu().numpy().tolist()
                    if min_dist_inf > 0:
                        bad = np.where(~np.isfinite(min_dist_batch))[0].tolist()
                        min_dist_inf_indices = [idx_np[i] for i in bad[:10]]
                    if min_dist_low_atoms > 0:
                        bad = np.where(atom_counts < 2)[0].tolist()
                        min_dist_low_atoms_indices = [idx_np[i] for i in bad[:10]]
                chol_log_clamp_rate = None
                chol_diag = None
                chol_min = (
                    model.cfg.model.chol_log_min_vec
                    if model.cfg.model.chol_log_min_vec is not None
                    else model.cfg.model.chol_log_min
                )
                chol_max = (
                    model.cfg.model.chol_log_max_vec
                    if model.cfg.model.chol_log_max_vec is not None
                    else model.cfg.model.chol_log_max
                )
                if model.cfg.model.cell_rep == "cholesky6" and (chol_min is not None or chol_max is not None):
                    diag = gram6_to_cholesky6(gram6, log_min=None, log_max=None)[:, :3]
                    diag_np = diag.detach().cpu().numpy()
                    chol_diag = diag.detach()
                    hit = np.zeros((diag_np.shape[0],), dtype=bool)
                    if chol_min is not None:
                        if isinstance(chol_min, (tuple, list)):
                            bound = np.asarray(chol_min, dtype=float).reshape((1, 3))
                            hit |= (diag_np <= bound + 1e-4).any(axis=1)
                        else:
                            hit |= diag_np.min(axis=1) <= float(chol_min) + 1e-4
                    if chol_max is not None:
                        if isinstance(chol_max, (tuple, list)):
                            bound = np.asarray(chol_max, dtype=float).reshape((1, 3))
                            hit |= (diag_np >= bound - 1e-4).any(axis=1)
                        else:
                            hit |= diag_np.max(axis=1) >= float(chol_max) - 1e-4
                    chol_log_clamp_rate = float(np.mean(hit)) if hit.size else 0.0
            msg = (
                f"[step {global_step}] loss={loss.item():.4f} "
                f"loss_f={metrics['loss_f'].item():.4f} "
                f"loss_g={metrics['loss_g'].item():.4f} "
                f"loss_z={metrics['loss_z'].item():.4f} "
                f"lr={lr:.6e}"
            )
            if use_geometry_fields:
                msg += (
                    f" loss_uv={metrics['loss_uv'].item():.4f}"
                    f" loss_zn={metrics['loss_zn'].item():.4f}"
                    f" loss_lat={metrics['loss_lat'].item():.4f}"
                )
            if use_thickness:
                msg += f" loss_t={metrics['loss_t'].item():.4f}"
            if "loss_min_dist" in metrics:
                msg += f" loss_min_dist={metrics['loss_min_dist'].item():.4f}"
            if "loss_comp" in metrics:
                msg += f" loss_comp={metrics['loss_comp'].item():.4f}"
            if "loss_vacuum" in metrics:
                msg += f" loss_vac={metrics['loss_vacuum'].item():.4f}"
            if "loss_cross_vacuum" in metrics:
                msg += f" loss_cv={metrics['loss_cross_vacuum'].item():.4f}"
            if "loss_angle" in metrics:
                msg += f" loss_angle={metrics['loss_angle'].item():.4f}"
            if "loss_cond_number" in metrics:
                msg += f" loss_cond_number={metrics['loss_cond_number'].item():.4f}"
            if "loss_volume" in metrics:
                msg += f" loss_vol={metrics['loss_volume'].item():.4f}"
            if "loss_c_len" in metrics:
                msg += f" loss_c={metrics['loss_c_len'].item():.4f}"
            if "loss_chol_bound" in metrics:
                msg += f" loss_chol={metrics['loss_chol_bound'].item():.4f}"
            if "loss_expand_collision" in metrics:
                msg += f" loss_expand={metrics['loss_expand_collision'].item():.4f}"
            if "pred_angle_out_rate" in metrics:
                msg += f" angle_out={metrics['pred_angle_out_rate'].item():.3f}"
            if "pred_cond_mean" in metrics:
                msg += f" cond_mean={metrics['pred_cond_mean'].item():.1f}"
            proxy_metrics = _compute_train_proxy_metrics(metrics, collision_rate)
            post_proj_proxy = _finite_metric(proxy_metrics.get("post_project_trigger_rate_train_proxy"))
            cond_proxy = _finite_metric(proxy_metrics.get("cond_violation_rate_train_proxy"))
            vac_proxy = _finite_metric(proxy_metrics.get("vacuum_violation_rate_train_proxy"))
            if post_proj_proxy is not None:
                msg += f" trigger_proxy={post_proj_proxy:.3f}"
            if cond_proxy is not None:
                msg += f" cond_proxy={cond_proxy:.3f}"
            if vac_proxy is not None:
                msg += f" vac_proxy={vac_proxy:.3f}"
            msg += (
                f" min_dist_mean={min_dist_mean:.3f} min_dist_p10={min_dist_p10:.3f}"
                f" collision_rate={collision_rate:.3f} min_dist_inf={min_dist_inf}"
                f" low_atoms={min_dist_low_atoms}"
            )
            if chol_log_clamp_rate is not None:
                msg += f" chol_log_clamp={chol_log_clamp_rate:.3f}"
            if "cross_vacuum_rate" in metrics:
                msg += f" cross_vac_rate={metrics['cross_vacuum_rate'].item():.3f}"
            if grad_norm is not None:
                msg += f" grad_norm={grad_norm:.3f}"
            if "s_f" in metrics:
                msg += (
                    f" s_f={metrics['s_f'].item():.3f}"
                    f" s_g={metrics['s_g'].item():.3f}"
                    f" s_z={metrics['s_z'].item():.3f}"
                )
                if use_geometry_fields:
                    msg += (
                        f" s_uv={metrics['s_uv'].item():.3f}"
                        f" s_zn={metrics['s_zn'].item():.3f}"
                        f" s_lat={metrics['s_lat'].item():.3f}"
                    )
                if use_thickness:
                    msg += f" s_t={metrics['s_t'].item():.3f}"
                if "s_comp" in metrics:
                    msg += f" s_comp={metrics['s_comp'].item():.3f}"
                if "s_vacuum" in metrics:
                    msg += f" s_vac={metrics['s_vacuum'].item():.3f}"
                if "s_cross_vacuum" in metrics:
                    msg += f" s_cv={metrics['s_cross_vacuum'].item():.3f}"
            print(msg)
            alerts = []
            if global_step < alert_steps:
                if collision_rate > alert_collision_rate:
                    alerts.append("collision_rate")
                if min_dist_p10 < alert_min_dist_p10:
                    alerts.append("min_dist_p10")
                if "vacuum_gap_mean" in metrics:
                    vac_gap_val = float(metrics["vacuum_gap_mean"].item())
                    if np.isfinite(vac_gap_val) and vac_gap_val > alert_vacuum_gap:
                        alerts.append("vacuum_gap")
                if chol_log_clamp_rate is not None and chol_log_clamp_rate > alert_chol_clamp_rate:
                    alerts.append("chol_log_clamp")
            if alerts:
                print(
                    f"[alert step {global_step}] "
                    + ", ".join(alerts)
                    + f" (collision_rate={collision_rate:.3f}, min_dist_p10={min_dist_p10:.3f})"
                )
            if metrics_log_path is not None:
                payload = {
                    "step": global_step,
                    "loss": float(loss.item()),
                    "loss_f": float(metrics["loss_f"].item()),
                    "loss_g": float(metrics["loss_g"].item()),
                    "loss_z": float(metrics["loss_z"].item()),
                    "loss_comp": float(metrics.get("loss_comp", torch.tensor(0.0)).item()),
                    "loss_vacuum": float(metrics.get("loss_vacuum", torch.tensor(0.0)).item()),
                    "loss_cross_vacuum": float(
                        metrics.get("loss_cross_vacuum", torch.tensor(0.0)).item()
                    ),
                    "lr": float(lr),
                    "min_dist_mean": min_dist_mean,
                    "min_dist_p10": min_dist_p10,
                    "collision_rate": collision_rate,
                    "min_dist_inf": min_dist_inf,
                    "min_dist_low_atoms": min_dist_low_atoms,
                }
                if min_dist_inf_indices is not None:
                    payload["min_dist_inf_indices"] = min_dist_inf_indices
                if min_dist_low_atoms_indices is not None:
                    payload["min_dist_low_atoms_indices"] = min_dist_low_atoms_indices
                if chol_log_clamp_rate is not None:
                    payload["chol_log_clamp_rate"] = float(chol_log_clamp_rate)
                if use_geometry_fields:
                    payload.update(
                        {
                            "loss_uv": float(metrics["loss_uv"].item()),
                            "loss_zn": float(metrics["loss_zn"].item()),
                            "loss_lat": float(metrics["loss_lat"].item()),
                        }
                    )
                if use_thickness:
                    payload["loss_t"] = float(metrics["loss_t"].item())
                if "loss_min_dist" in metrics:
                    payload["loss_min_dist"] = float(metrics["loss_min_dist"].item())
                if "loss_angle" in metrics:
                    payload["loss_angle"] = float(metrics["loss_angle"].item())
                if "loss_cond_number" in metrics:
                    payload["loss_cond_number"] = float(metrics["loss_cond_number"].item())
                if "loss_volume" in metrics:
                    payload["loss_volume"] = float(metrics["loss_volume"].item())
                if "loss_c_len" in metrics:
                    payload["loss_c_len"] = float(metrics["loss_c_len"].item())
                if "loss_anisotropy" in metrics:
                    payload["loss_anisotropy"] = float(metrics["loss_anisotropy"].item())
                if "loss_chol_bound" in metrics:
                    payload["loss_chol_bound"] = float(metrics["loss_chol_bound"].item())
                if "loss_expand_collision" in metrics:
                    payload["loss_expand_collision"] = float(metrics["loss_expand_collision"].item())
                if "pred_angle_out_rate" in metrics:
                    payload["pred_angle_out_rate"] = float(metrics["pred_angle_out_rate"].item())
                if "pred_cond_mean" in metrics:
                    payload["pred_cond_mean"] = float(metrics["pred_cond_mean"].item())
                if "cond_valid_rate" in metrics:
                    payload["cond_valid_rate"] = float(metrics["cond_valid_rate"].item())
                if "cond_gram_mean" in metrics:
                    payload.update(
                        {
                            "cond_gram_mean": float(metrics["cond_gram_mean"].item()),
                            "cond_gram_p50": float(metrics["cond_gram_p50"].item()),
                            "cond_gram_p95": float(metrics["cond_gram_p95"].item()),
                            "cond_gram_max": float(metrics["cond_gram_max"].item()),
                            "cond_gram_violation_rate": float(
                                metrics["cond_gram_violation_rate"].item()
                            ),
                        }
                    )
                if "cond_lattice_mean" in metrics:
                    payload.update(
                        {
                            "cond_lattice_mean": float(metrics["cond_lattice_mean"].item()),
                            "cond_lattice_p50": float(metrics["cond_lattice_p50"].item()),
                            "cond_lattice_p95": float(metrics["cond_lattice_p95"].item()),
                            "cond_lattice_max": float(metrics["cond_lattice_max"].item()),
                            "cond_lattice_violation_rate": float(
                                metrics["cond_lattice_violation_rate"].item()
                            ),
                        }
                    )
                if "cond_gram_lattice_spearman" in metrics:
                    payload["cond_gram_lattice_spearman"] = float(
                        metrics["cond_gram_lattice_spearman"].item()
                    )
                if "cond_diff_abs_mean" in metrics:
                    payload.update(
                        {
                            "cond_diff_abs_mean": float(metrics["cond_diff_abs_mean"].item()),
                            "cond_diff_abs_p95": float(metrics["cond_diff_abs_p95"].item()),
                            "cond_diff_rel_mean": float(metrics["cond_diff_rel_mean"].item()),
                            "cond_diff_rel_p95": float(metrics["cond_diff_rel_p95"].item()),
                        }
                    )
                if grad_prefix_norms:
                    for prefix, value in grad_prefix_norms.items():
                        key = f"grad_{prefix.replace('.', '_')}"
                        payload[key] = value
                if "chol_bound_rate" in metrics:
                    payload["chol_bound_rate"] = float(metrics["chol_bound_rate"].item())
                if "min_dist_pred_mean" in metrics:
                    payload["min_dist_pred_mean"] = float(metrics["min_dist_pred_mean"].item())
                if "min_dist_pred_p10" in metrics:
                    payload["min_dist_pred_p10"] = float(metrics["min_dist_pred_p10"].item())
                if "vacuum_gap_mean" in metrics:
                    payload["vacuum_gap_mean"] = float(metrics["vacuum_gap_mean"].item())
                if "cross_vacuum_rate" in metrics:
                    payload["cross_vacuum_rate"] = float(metrics["cross_vacuum_rate"].item())
                if "c_len_mean" in metrics:
                    payload["c_len_mean"] = float(metrics["c_len_mean"].item())
                if "thickness_gap_mean" in metrics:
                    payload["thickness_gap_mean"] = float(metrics["thickness_gap_mean"].item())
                if "lengths_std_mean" in metrics:
                    lengths_std_mean = float(metrics["lengths_std_mean"].item())
                    if math.isfinite(lengths_std_mean):
                        payload["lengths_std_mean"] = lengths_std_mean
                if "pred_x0_f_mean" in metrics:
                    payload.update(
                        {
                            "pred_x0_f_mean": float(metrics["pred_x0_f_mean"].item()),
                            "pred_x0_f_std": float(metrics["pred_x0_f_std"].item()),
                            "pred_v_f_mean": float(metrics["pred_v_f_mean"].item()),
                            "pred_v_f_std": float(metrics["pred_v_f_std"].item()),
                        }
                    )
                if "s_f" in metrics:
                    payload.update(
                        {
                            "s_f": float(metrics["s_f"].item()),
                            "s_g": float(metrics["s_g"].item()),
                            "s_z": float(metrics["s_z"].item()),
                        }
                    )
                    if use_geometry_fields:
                        payload.update(
                            {
                                "s_uv": float(metrics["s_uv"].item()),
                                "s_zn": float(metrics["s_zn"].item()),
                                "s_lat": float(metrics["s_lat"].item()),
                            }
                        )
                    if use_thickness:
                        payload["s_t"] = float(metrics["s_t"].item())
                    if "s_angle" in metrics:
                        payload["s_angle"] = float(metrics["s_angle"].item())
                    if "s_cross_vacuum" in metrics:
                        payload["s_cross_vacuum"] = float(metrics["s_cross_vacuum"].item())
                if "s_cond" in metrics:
                    payload["s_cond"] = float(metrics["s_cond"].item())
                if grad_norm is not None:
                    payload["grad_norm"] = float(grad_norm)
                payload.update(proxy_metrics)
                if alerts:
                    payload["alerts"] = alerts
                if loss_weight_state is not None:
                    payload["lambda_vacuum"] = float(loss_weight_state.get("vacuum", 0.0))
                    payload["lambda_cross_vacuum"] = float(
                        loss_weight_state.get("cross_vacuum", 0.0)
                    )
                    payload["lambda_cond"] = float(loss_weight_state.get("cond", 0.0))
                    payload["lambda_chol_bound"] = float(loss_weight_state.get("chol_bound", 0.0))
                    payload["lambda_expand_collision"] = float(
                        loss_weight_state.get("expand_collision", 0.0)
                    )
                    payload["lambda_volume"] = float(loss_weight_state.get("volume", 0.0))
                    payload["lambda_c_len"] = float(loss_weight_state.get("c_len", 0.0))
                    payload["lambda_anisotropy"] = float(loss_weight_state.get("anisotropy", 0.0))
                with metrics_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
            if tb_writer is not None and global_step % tb_interval == 0:
                tb_writer.add_scalar("loss/total", float(loss.item()), global_step)
                tb_writer.add_scalar("loss/f", float(metrics["loss_f"].item()), global_step)
                tb_writer.add_scalar("loss/g", float(metrics["loss_g"].item()), global_step)
                tb_writer.add_scalar("loss/z", float(metrics["loss_z"].item()), global_step)
                if "loss_vacuum" in metrics:
                    tb_writer.add_scalar(
                        "loss/vacuum", float(metrics["loss_vacuum"].item()), global_step
                    )
                if "loss_cross_vacuum" in metrics:
                    tb_writer.add_scalar(
                        "loss/cross_vacuum", float(metrics["loss_cross_vacuum"].item()), global_step
                    )
                if "loss_min_dist" in metrics:
                    tb_writer.add_scalar(
                        "loss/min_dist", float(metrics["loss_min_dist"].item()), global_step
                    )
                if grad_norm is not None:
                    tb_writer.add_scalar("grad/total_norm", float(grad_norm), global_step)
                tb_writer.add_scalar("dist/min_dist_mean", min_dist_mean, global_step)
                tb_writer.add_scalar("dist/min_dist_p10", min_dist_p10, global_step)
                tb_writer.add_scalar("dist/collision_rate", collision_rate, global_step)
                if "vacuum_gap_mean" in metrics:
                    tb_writer.add_scalar(
                        "vacuum/gap_mean", float(metrics["vacuum_gap_mean"].item()), global_step
                    )
                if "cross_vacuum_rate" in metrics:
                    tb_writer.add_scalar(
                        "vacuum/cross_rate", float(metrics["cross_vacuum_rate"].item()), global_step
                    )
                if chol_log_clamp_rate is not None:
                    tb_writer.add_scalar(
                        "lattice/chol_log_clamp_rate", float(chol_log_clamp_rate), global_step
                    )
                if valid_mask.any():
                    tb_writer.add_histogram(
                        "dist/min_dist",
                        torch.tensor(min_dist_batch[valid_mask], dtype=torch.float32),
                        global_step,
                    )
                vac_gap = metrics.get("vacuum_gap")
                if vac_gap is not None:
                    vac_gap = vac_gap.detach().cpu()
                    vac_gap = vac_gap[torch.isfinite(vac_gap)]
                    if vac_gap.numel() > 0:
                        tb_writer.add_histogram("vacuum/gap", vac_gap, global_step)
                if chol_diag is not None:
                    tb_writer.add_histogram(
                        "lattice/chol_diag", chol_diag.detach().cpu(), global_step
                    )
            if wandb_run is not None:
                wandb_payload = {
                    "loss/total": float(loss.item()),
                    "loss/f": float(metrics["loss_f"].item()),
                    "loss/g": float(metrics["loss_g"].item()),
                    "loss/z": float(metrics["loss_z"].item()),
                    "dist/min_dist_mean": min_dist_mean,
                    "dist/min_dist_p10": min_dist_p10,
                    "dist/collision_rate": collision_rate,
                }
                if "loss_vacuum" in metrics:
                    wandb_payload["loss/vacuum"] = float(metrics["loss_vacuum"].item())
                if "loss_cross_vacuum" in metrics:
                    wandb_payload["loss/cross_vacuum"] = float(
                        metrics["loss_cross_vacuum"].item()
                    )
                if "loss_min_dist" in metrics:
                    wandb_payload["loss/min_dist"] = float(metrics["loss_min_dist"].item())
                if "vacuum_gap_mean" in metrics:
                    wandb_payload["vacuum/gap_mean"] = float(metrics["vacuum_gap_mean"].item())
                if "cross_vacuum_rate" in metrics:
                    wandb_payload["vacuum/cross_rate"] = float(
                        metrics["cross_vacuum_rate"].item()
                    )
                if chol_log_clamp_rate is not None:
                    wandb_payload["lattice/chol_log_clamp_rate"] = float(chol_log_clamp_rate)
                if grad_norm is not None:
                    wandb_payload["grad/total_norm"] = float(grad_norm)
                if alerts:
                    wandb_payload["alerts"] = ",".join(alerts)
                wandb_run.log(wandb_payload, step=global_step)

        global_step += 1
        total_loss += loss.item()
        step_count += 1

    mean_loss = total_loss / max(step_count, 1)
    return global_step, mean_loss


def save_checkpoint(save_path: Path, payload: dict) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def _serialize_args(args: argparse.Namespace) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _serialize_cond_stats(cond_stats: dict | None) -> dict:
    if not cond_stats:
        return {}
    payload: dict[str, list | float] = {}
    for key, value in cond_stats.items():
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().tolist()
        else:
            payload[key] = float(value)
    return payload


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _compute_train_proxy_metrics(metrics: dict[str, torch.Tensor], collision_rate: float) -> dict[str, float]:
    cond_violation = _finite_metric(
        metrics.get("cond_lattice_violation_rate", torch.tensor(float("nan"))).item()
        if "cond_lattice_violation_rate" in metrics
        else None
    )
    if cond_violation is None:
        cond_violation = _finite_metric(
            metrics.get("cond_gram_violation_rate", torch.tensor(float("nan"))).item()
            if "cond_gram_violation_rate" in metrics
            else None
        )

    vacuum_violation = None
    vac_gap = metrics.get("vacuum_gap")
    if isinstance(vac_gap, torch.Tensor):
        with torch.no_grad():
            gap = vac_gap.detach().reshape(-1)
            finite = torch.isfinite(gap)
            if bool(finite.any()):
                vacuum_violation = float((gap[finite] > 0.0).float().mean().item())
    if vacuum_violation is None:
        vac_gap_mean = _finite_metric(
            metrics.get("vacuum_gap_mean", torch.tensor(float("nan"))).item()
            if "vacuum_gap_mean" in metrics
            else None
        )
        if vac_gap_mean is not None:
            vacuum_violation = 1.0 if vac_gap_mean > 0.0 else 0.0

    angle_violation = _finite_metric(
        metrics.get("pred_angle_out_rate", torch.tensor(float("nan"))).item()
        if "pred_angle_out_rate" in metrics
        else None
    )
    collision_proxy = _finite_metric(collision_rate)
    parts = [v for v in [cond_violation, vacuum_violation, angle_violation, collision_proxy] if v is not None]
    trigger_proxy = max(parts) if parts else float("nan")
    return {
        "post_project_trigger_rate_train_proxy": float(trigger_proxy),
        "cond_violation_rate_train_proxy": float(cond_violation) if cond_violation is not None else float("nan"),
        "vacuum_violation_rate_train_proxy": float(vacuum_violation) if vacuum_violation is not None else float("nan"),
    }


def _aggregate_train_metrics_jsonl(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return {"available": False, "reason": "train_metrics.jsonl not found"}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return {"available": False, "reason": "train_metrics.jsonl has no valid rows"}

    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            fv = _finite_metric(value)
            if fv is None:
                continue
            metrics.setdefault(key, []).append(fv)

    summary: dict[str, Any] = {
        "available": True,
        "rows": len(rows),
        "first_step": int(rows[0].get("step", 0)),
        "last_step": int(rows[-1].get("step", 0)),
        "metrics": {},
    }
    for key, values in sorted(metrics.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        summary["metrics"][key] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50.0)),
            "p95": float(np.percentile(arr, 95.0)),
        }

    proxy_keys = (
        "post_project_trigger_rate_train_proxy",
        "cond_violation_rate_train_proxy",
        "vacuum_violation_rate_train_proxy",
    )
    trend: dict[str, Any] = {}
    for key in proxy_keys:
        seq = [v for v in metrics.get(key, []) if math.isfinite(v)]
        if len(seq) < 2:
            trend[key] = {
                "available": False,
                "reason": "insufficient_points",
                "first_half_mean": None,
                "second_half_mean": None,
                "delta_second_minus_first": None,
                "improved": None,
            }
            continue
        split = max(1, len(seq) // 2)
        first = float(np.mean(np.asarray(seq[:split], dtype=float)))
        second = float(np.mean(np.asarray(seq[split:], dtype=float)))
        delta = second - first
        trend[key] = {
            "available": True,
            "first_half_mean": first,
            "second_half_mean": second,
            "delta_second_minus_first": delta,
            "improved": bool(second < first),
        }
    summary["proxy_trend"] = trend
    return summary


def _resolve_model_hparams(args: argparse.Namespace) -> dict:
    overrides = {
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "time_embed_dim": args.time_embed_dim,
        "z_embed_dim": args.z_embed_dim,
        "f_embed_dim": args.f_embed_dim,
        "rbf_dim": args.rbf_dim,
        "pair_mlp_hidden": args.pair_mlp_hidden,
    }
    return resolve_model_hparams(args.model_size, overrides)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed, args.deterministic)
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.npz is None:
        raise ValueError("--npz is required; CSV-based datasets are no longer supported.")
    split_indices: list[int] | None = None
    if args.coord_frame == "canon" and not args.align_atoms:
        raise ValueError("--coord-frame canon requires --align-atoms to keep per-atom fields aligned.")
    dataset = C2DBTokenNPZDataset(
        args.npz, align_atoms=args.align_atoms, coord_frame=args.coord_frame
    )
    if args.split_json is not None and args.split != "all":
        split_payload = load_c2db_split(args.split_json)
        split_indices = select_split_indices(split_payload, args.split)
        if not split_indices:
            raise ValueError(f"Split subset {args.split!r} is empty in {args.split_json}.")
        validate_split_indices(split_indices, total=len(dataset))
    if args.g_scale != dataset.g_scale:
        print(f"[warn] g_scale {args.g_scale} != dataset g_scale {dataset.g_scale}")
    g_scale = dataset.g_scale
    if getattr(dataset, "coord_frame", None) is not None or getattr(dataset, "schema_version", None) is not None:
        print(
            "[info] dataset alignment: "
            f"align_atoms={dataset.align_atoms} "
            f"coord_frame={getattr(dataset, 'coord_frame', None)} "
            f"schema_version={getattr(dataset, 'schema_version', None)}"
        )
    if getattr(dataset, "coord_frame_actual", None) is not None and dataset.coord_frame_actual != args.coord_frame:
        print(
            f"[warn] requested coord_frame={args.coord_frame} "
            f"but dataset fell back to coord_frame={dataset.coord_frame_actual}"
        )
    dataset = IndexedDataset(dataset)

    quality_map: dict[str, dict[str, Any]] | None = None
    quality_filter_summary: dict[str, Any] | None = None
    if args.quality_jsonl is not None and not args.quality_jsonl.exists():
        print(f"[warn] quality-jsonl not found at {args.quality_jsonl}; skipping quality filter.")
        args.quality_jsonl = None
    if args.quality_jsonl is not None:
        quality_map = _load_quality_map(args.quality_jsonl)
        if not quality_map:
            raise ValueError(f"No quality records found in {args.quality_jsonl}")
    coord_frame_actual = getattr(dataset, "coord_frame_actual", args.coord_frame)

    base_indices: list[int] | None = split_indices
    if quality_map is not None:
        indices = base_indices if base_indices is not None else list(range(len(dataset)))
        buckets = _parse_quality_buckets(args.quality_buckets)
        filtered_indices, missing, filtered_out = _filter_indices_by_quality(
            dataset=dataset,
            indices=indices,
            quality_map=quality_map,
            allowed_buckets=buckets,
            require_hard_pass=args.quality_hard_pass_only,
        )
        before = len(indices)
        print(
            "[info] quality filter: "
            f"kept {len(filtered_indices)}/{before} rows "
            f"(missing={missing}, filtered={filtered_out})"
        )
        if not filtered_indices:
            raise ValueError("Quality filter removed all usable training samples.")
        base_indices = filtered_indices
        quality_filter_summary = {
            "kept": len(filtered_indices),
            "total": before,
            "missing": missing,
            "filtered_out": filtered_out,
            "buckets": buckets,
            "hard_pass_only": args.quality_hard_pass_only,
            "quality_jsonl": str(args.quality_jsonl),
        }
    min_dist_all: torch.Tensor | None = None
    needs_min_dist = args.curriculum_collision or (args.filter_min_dist_below is not None)
    if needs_min_dist:
        min_dist_all = _compute_dataset_min_dist(
            dataset,
            pbc_mask=pbc_mask,
            g_scale=g_scale,
            batch_size=min(512, max(32, args.batch_size)),
        )
        if min_dist_all.numel() != len(dataset):
            raise ValueError(
                f"min_dist computation returned shape {tuple(min_dist_all.shape)} but dataset has len={len(dataset)}"
            )
        if base_indices is None:
            base_indices = list(range(len(dataset)))
        if args.filter_min_dist_below is not None:
            cut = float(args.filter_min_dist_below)
            before = len(base_indices)
            base_indices = [i for i in base_indices if float(min_dist_all[i]) >= cut]
            print(f"[info] filter-min-dist-below={cut:.3f}: kept {len(base_indices)}/{before} samples")
        if args.curriculum_collision:
            easy = sum(float(min_dist_all[i]) >= float(args.curriculum_min_dist_cut) for i in base_indices)
            hard = len(base_indices) - easy
            print(
                "[info] collision curriculum enabled: "
                f"easy={easy} hard={hard} "
                f"(cut={float(args.curriculum_min_dist_cut):.3f}, ramp_epochs={args.curriculum_epochs})"
            )

    volume_stats: tuple[float, float, float] | None = None
    if args.auto_volume_bounds:
        dataset_raw = _unwrap_indexed_dataset(dataset)
        volume_stats = _estimate_volume_stats(dataset_raw, g_scale, indices=base_indices)
        if volume_stats is None:
            print("[warn] auto-volume-bounds: unable to compute volume stats; keeping defaults.")
        else:
            v_min, v_med, v_max = volume_stats
            args.volume_min = float(v_min)
            args.volume_max = float(v_max)
            print(
                "[info] auto-volume-bounds: "
                f"p1={v_min:.3f} p50={v_med:.3f} p99={v_max:.3f} "
                f"-> volume_min={args.volume_min:.3f} volume_max={args.volume_max:.3f}"
            )

    use_condition = args.use_condition
    geom_available = all(
        getattr(dataset, name, None) is not None for name in ("uv_angle", "z_norm", "lattice_param")
    )
    use_geometry_fields = args.use_geometry_fields and geom_available and coord_frame_actual == "canon"
    if args.use_geometry_fields and not geom_available:
        print("[warn] geometry fields not found in dataset; disabling geometry heads.")
    if args.use_geometry_fields and coord_frame_actual != "canon":
        print(
            "[warn] dataset coord_frame_actual=%r != 'canon'; disabling geometry heads."
            % coord_frame_actual
        )
    if args.use_geometry_fields and coord_frame_actual == "canon":
        coord_frame_meta = getattr(dataset, "coord_frame", None)
        if coord_frame_meta is not None and str(coord_frame_meta) != "canon":
            print(
                "[warn] dataset coord_frame=%r != 'canon' while coord_frame_actual is 'canon'; "
                "disabling geometry heads to avoid mixed frames."
                % coord_frame_meta
            )
            use_geometry_fields = False
    if (
        use_geometry_fields
        and isinstance(dataset, C2DBTokenNPZDataset)
        and getattr(dataset, "order_idx", None) is not None
        and not dataset.align_atoms
    ):
        raise ValueError(
            "Geometry fields enabled but align_atoms is False while order_idx exists. "
            "Set --align-atoms or disable geometry heads."
        )
    t_available = getattr(dataset, "t", None) is not None
    use_thickness = use_geometry_fields and t_available
    if use_geometry_fields and not t_available:
        print("[warn] thickness field not found in dataset; disabling t head.")
    t_stats = None
    if use_thickness:
        t_values = getattr(dataset, "t", None)
        if t_values is not None:
            t_float = t_values.float()
            if split_indices is not None:
                t_float = t_float.index_select(0, torch.as_tensor(split_indices, dtype=torch.long))
            t_mean = float(t_float.mean().item())
            t_std = float(t_float.std(unbiased=False).clamp_min(1e-6).item())
            if t_std <= 0:
                t_std = 1e-6
            t_stats = (t_mean, t_std)
        else:
            print("[warn] thickness field missing for normalization; using raw t.")
    z_norm_stats = None
    if use_geometry_fields:
        z_norm_values = getattr(dataset, "z_norm", None)
        if z_norm_values is not None:
            z_norm_float = z_norm_values.float()
            if split_indices is not None:
                z_norm_float = z_norm_float.index_select(
                    0, torch.as_tensor(split_indices, dtype=torch.long)
                )
            z_mean = float(z_norm_float.mean().item())
            z_std = float(z_norm_float.std(unbiased=False).clamp_min(1e-6).item())
            z_norm_stats = (z_mean, z_std)
    cond_max_atoms = int(getattr(dataset, "max_atoms", args.max_atoms))
    cond_dim = 0
    cond_stats = {}
    cond_fields = _resolve_cond_fields(args)
    normalize_fields = _parse_cond_fields(args.cond_normalize_fields)
    element_ids = _parse_element_ids(args.element_ids)
    if use_condition:
        sample = dataset[split_indices[0]] if split_indices is not None else dataset[0]
        cond_dim = _infer_cond_dim(sample, cond_fields, num_elements=118)
        cond_stats = _compute_cond_stats(
            dataset, cond_fields, normalize_fields, indices=split_indices, t_stats=t_stats
        )

    lattice_stats = None
    if args.cell_rep == "cholesky6":
        s10, s50, s90, log_std = _estimate_scube_stats(dataset, g_scale, indices=split_indices)
        if args.cell_init == "iso" and args.cell_init_scale is None:
            args.cell_init_scale = args.cell_init_scale_factor * s50
        if args.cell_init_noise is None:
            args.cell_init_noise = float(min(max(log_std, 0.1), 0.2))
        if args.chol_log_min is None:
            args.chol_log_min = float(np.log(max(args.cell_log_min_factor * s10, 1e-6)))
        if args.chol_log_max is None:
            args.chol_log_max = float(np.log(max(args.cell_log_max_factor * s90, 1e-6)))

        # Root-cause fix for slab collapse:
        # Use per-dimension Cholesky-diagonal bounds instead of a single scalar bound shared across axes.
        chol_diag_stats = None
        dataset_raw = _unwrap_indexed_dataset(dataset)
        if isinstance(dataset_raw, C2DBTokenNPZDataset):
            lattice = None
            coord_frame_actual = getattr(dataset_raw, "coord_frame_actual", args.coord_frame)
            if coord_frame_actual == "canon" and getattr(dataset_raw, "lattice_canon", None) is not None:
                lattice = dataset_raw.lattice_canon.float()
            elif getattr(dataset_raw, "lattice", None) is not None:
                lattice = dataset_raw.lattice.float()
            if lattice is not None and base_indices is not None:
                lattice = lattice.index_select(0, torch.as_tensor(base_indices, dtype=torch.long))
            if lattice is not None:
                chol_diag_stats = _estimate_chol_diag_stats_from_lattice(lattice, g_scale)
        if chol_diag_stats is not None:
            diag10, diag50, diag90, diag_log_std = chol_diag_stats
            shift_min = math.log(max(float(args.cell_log_min_factor), 1e-12))
            shift_max = math.log(max(float(args.cell_log_max_factor), 1e-12))
            chol_log_min_vec = tuple(float(v) + shift_min for v in diag10)
            chol_log_max_vec = tuple(float(v) + shift_max for v in diag90)
            args.chol_log_min_vec = chol_log_min_vec
            args.chol_log_max_vec = chol_log_max_vec
            print(
                "[info] chol_diag bounds "
                f"min_vec={chol_log_min_vec} max_vec={chol_log_max_vec} "
                f"(diag_p10={diag10}, diag_p50={diag50}, diag_p90={diag90}, diag_log_std={diag_log_std:.4f})"
            )
        else:
            args.chol_log_min_vec = None
            args.chol_log_max_vec = None
        side_min = None
        side_max = None
        if args.chol_log_min is not None:
            side_min = math.exp(float(args.chol_log_min)) * math.sqrt(g_scale)
        if args.chol_log_max is not None:
            side_max = math.exp(float(args.chol_log_max)) * math.sqrt(g_scale)
        print(
            "[info] scube stats "
            f"s10={s10:.4f} s50={s50:.4f} s90={s90:.4f} log_std={log_std:.4f}"
        )
        print(
            "[info] chol_log bounds "
            f"min={args.chol_log_min:.4f} max={args.chol_log_max:.4f} "
            f"side_min={side_min:.3f} side_max={side_max:.3f}"
        )
        lattice_stats = {
            "scube_p10": float(s10),
            "scube_p50": float(s50),
            "scube_p90": float(s90),
            "scube_log_std": float(log_std),
            "chol_log_min": float(args.chol_log_min),
            "chol_log_max": float(args.chol_log_max),
            "side_min": float(side_min) if side_min is not None else None,
            "side_max": float(side_max) if side_max is not None else None,
            "volume_min": float(side_min**3) if side_min is not None else None,
            "volume_max": float(side_max**3) if side_max is not None else None,
        }
        if args.volume_max is None and lattice_stats["volume_max"] is not None:
            args.volume_max = float(lattice_stats["volume_max"])

    if args.dual_graph and args.edge_type_dim == 0:
        args.edge_type_dim = 4
        print("[warn] --dual-graph enabled with edge_type_dim=0; defaulting to 4.")

    dataset_len = len(base_indices) if base_indices is not None else len(dataset)
    if not args.drop_last and dataset_len < args.batch_size:
        print("[warn] --no-drop-last with dataset smaller than batch size; expect tiny batches.")

    def _make_loader(epoch_indices: list[int] | None) -> DataLoader:
        return prepare_dataloader(
            dataset,
            args.batch_size,
            args.num_workers,
            args.bucket_batches,
            args.bucket_shuffle,
            seed=args.seed,
            drop_last=args.drop_last,
            indices=epoch_indices,
        )

    loader = _make_loader(base_indices)

    model_hparams = _resolve_model_hparams(args)
    model_cfg = AtomTransformerConfig(
        embed_dim=model_hparams["embed_dim"],
        depth=model_hparams["depth"],
        num_heads=model_hparams["num_heads"],
        mlp_ratio=model_hparams["mlp_ratio"],
        dropout=model_hparams["dropout"],
        time_embed_dim=model_hparams["time_embed_dim"],
        z_embed_dim=model_hparams["z_embed_dim"],
        f_embed_dim=model_hparams["f_embed_dim"],
        rbf_dim=model_hparams["rbf_dim"],
        pair_mlp_hidden=model_hparams["pair_mlp_hidden"],
        num_elements=118,
        k_neighbors=args.k_neighbors,
        g_scale=g_scale,
        cell_rep=args.cell_rep,
        chol_log_min=args.chol_log_min,
        chol_log_max=args.chol_log_max,
        chol_log_min_vec=getattr(args, "chol_log_min_vec", None),
        chol_log_max_vec=getattr(args, "chol_log_max_vec", None),
        cond_dim=cond_dim,
        use_comp_encoder=args.use_comp_encoder,
        comp_embed_dim=args.comp_embed_dim,
        comp_pool_mode=args.comp_pool_mode,
        comp_use_frac=args.comp_use_frac,
        element_ids=element_ids,
        pbc_mask=pbc_mask,
        dual_graph=args.dual_graph,
        edge_type_dim=args.edge_type_dim,
        edge_type_gating=args.edge_type_gating,
        wrap_embed_dim=args.wrap_embed_dim,
        tail_adapter=args.tail_adapter,
        tail_hidden_dim=args.tail_hidden_dim,
        tail_scale=args.tail_scale,
    )
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    denoiser_cfg.diffusion.mode = args.mode
    denoiser_cfg.diffusion.cell_rep = args.cell_rep
    denoiser_cfg.diffusion.chol_log_min = args.chol_log_min
    denoiser_cfg.diffusion.chol_log_max = args.chol_log_max
    denoiser_cfg.diffusion.chol_log_min_vec = getattr(args, "chol_log_min_vec", None)
    denoiser_cfg.diffusion.chol_log_max_vec = getattr(args, "chol_log_max_vec", None)
    denoiser_cfg.diffusion.lambda_comp = float(args.comp_loss_weight)
    denoiser_cfg.diffusion.comp_loss_mode = str(args.comp_loss_mode)
    denoiser_cfg.diffusion.lambda_vacuum = float(args.vacuum_loss_weight)
    denoiser_cfg.diffusion.vacuum_min = float(args.vacuum_min)
    denoiser_cfg.diffusion.vacuum_loss_power = int(args.vacuum_loss_power)
    denoiser_cfg.diffusion.vacuum_loss_mode = str(args.vacuum_loss_mode)
    denoiser_cfg.diffusion.lambda_cross_vacuum = float(args.cross_vacuum_loss_weight)
    denoiser_cfg.diffusion.cross_vacuum_bond_cut = float(args.cross_vacuum_bond_cut)
    denoiser_cfg.diffusion.cross_vacuum_power = int(args.cross_vacuum_power)
    symmetry_mode = str(args.symmetry_mode).lower()
    symmetry_loss_weight = float(args.symmetry_loss_weight)
    if symmetry_mode == "off":
        symmetry_loss_weight = 0.0
    denoiser_cfg.symmetry_loss_weight = symmetry_loss_weight
    denoiser_cfg.symmetry_symprec = float(args.symmetry_symprec)
    denoiser_cfg.diffusion.lambda_angle = float(args.angle_loss_weight)
    denoiser_cfg.diffusion.angle_min = float(args.angle_min)
    denoiser_cfg.diffusion.angle_max = float(args.angle_max)
    denoiser_cfg.diffusion.angle_param_mode = str(args.angle_param_mode)
    denoiser_cfg.diffusion.angle_sigmoid_tau = float(args.angle_sigmoid_tau)
    denoiser_cfg.diffusion.lambda_cond = float(args.cond_loss_weight)
    denoiser_cfg.diffusion.cond_max = float(args.cond_max)
    denoiser_cfg.diffusion.lambda_volume = float(args.volume_loss_weight)
    denoiser_cfg.diffusion.volume_min = float(args.volume_min)
    denoiser_cfg.diffusion.volume_max = args.volume_max
    denoiser_cfg.diffusion.lambda_c_len = float(args.c_len_loss_weight)
    denoiser_cfg.diffusion.c_len_min = float(args.c_len_min)
    denoiser_cfg.diffusion.lambda_anisotropy = float(args.anisotropy_loss_weight)
    denoiser_cfg.diffusion.anisotropy_min_std = float(args.anisotropy_min_std)
    denoiser_cfg.diffusion.loss_hinge = str(args.loss_hinge)
    denoiser_cfg.diffusion.loss_softplus_beta = float(args.loss_softplus_beta)
    denoiser_cfg.diffusion.lambda_chol_bound = float(args.chol_bound_loss_weight)
    denoiser_cfg.diffusion.chol_bound_margin = float(args.chol_bound_margin)
    denoiser_cfg.diffusion.chol_bound_power = int(args.chol_bound_power)
    denoiser_cfg.diffusion.lambda_expand_collision = float(args.expand_on_collision_weight)
    denoiser_cfg.diffusion.expand_min_dist_cut = float(args.expand_on_collision_cut)
    denoiser_cfg.diffusion.noise_scale_zn = float(args.noise_scale_zn)
    denoiser_cfg.diffusion.cell_init = args.cell_init
    denoiser_cfg.diffusion.cell_init_scale = args.cell_init_scale
    denoiser_cfg.diffusion.cell_init_noise = args.cell_init_noise
    if use_condition:
        denoiser_cfg.diffusion.cond_drop_prob = args.cond_drop_prob
    denoiser_cfg.diffusion.use_uncertainty_weighting = not args.no_uncertainty_weighting
    denoiser_cfg.min_dist_train_cut = float(args.min_dist_train_cut)
    denoiser_cfg.min_dist_train_weight = float(args.min_dist_train_weight)
    denoiser_cfg.chol_log_relax = float(args.chol_log_relax)
    model = AtomDenoiser(denoiser_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.3f}M (model_size={args.model_size})")
    print("[info] pred_target=x0")
    betas = _parse_betas(args.betas)
    param_groups = _build_param_groups(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=betas)
    ema = EMA(model, args.ema_decay) if args.ema else None
    loss_weight_scheduler = None
    base_loss_weights = {
        "vacuum": float(args.vacuum_loss_weight),
        "cross_vacuum": float(args.cross_vacuum_loss_weight),
        "cond": float(args.cond_loss_weight),
        "chol_bound": float(args.chol_bound_loss_weight),
        "expand_collision": float(args.expand_on_collision_weight),
        "volume": float(args.volume_loss_weight),
        "c_len": float(args.c_len_loss_weight),
        "anisotropy": float(args.anisotropy_loss_weight),
    }
    warmup_keys = _parse_loss_weight_keys(args.loss_weight_warmup_keys)
    if args.loss_weight_warmup_steps > 0 and warmup_keys:
        schedule_cfg = LossWeightScheduleConfig(
            warmup_steps=int(args.loss_weight_warmup_steps),
            start_factor=float(args.loss_weight_warmup_start),
            end_factor=float(args.loss_weight_warmup_end),
            keys=tuple(warmup_keys),
            schedule=str(args.loss_weight_schedule),
        )
        loss_weight_scheduler = LossWeightScheduler(base_loss_weights, schedule_cfg)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = collect_run_metadata(argv=sys.argv)
    config_payload = {
        "created_at": run_stamp,
        "run_metadata": run_metadata,
        "args": _serialize_args(args),
        "pred_target": "x0",
        "model_config": asdict(model_cfg),
        "diffusion_config": asdict(denoiser_cfg.diffusion),
        "optimizer_config": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": betas,
            "warmup_steps": args.warmup_steps,
            "min_lr": args.min_lr,
            "lr_schedule": args.lr_schedule,
            "clip_grad": args.clip_grad,
            "ema": args.ema,
            "ema_decay": args.ema_decay,
        },
        "loss_weight_schedule": {
            "warmup_steps": int(args.loss_weight_warmup_steps),
            "keys": _parse_loss_weight_keys(args.loss_weight_warmup_keys),
            "start_factor": float(args.loss_weight_warmup_start),
            "end_factor": float(args.loss_weight_warmup_end),
            "schedule": str(args.loss_weight_schedule),
        },
        "volume_c_len_warmup": {
            "warmup_steps": int(args.volume_c_len_warmup_steps),
            "start_factor": float(args.volume_c_len_warmup_start),
            "end_factor": float(args.volume_c_len_warmup_end),
        },
        "volume_stats": {
            "p1": float(volume_stats[0]) if volume_stats is not None else None,
            "p50": float(volume_stats[1]) if volume_stats is not None else None,
            "p99": float(volume_stats[2]) if volume_stats is not None else None,
        },
        "denoiser_config": {
            "chol_log_relax": float(denoiser_cfg.chol_log_relax),
        },
        "cond_config": {
            "use_condition": use_condition,
            "cond_dim": cond_dim,
            "max_atoms": cond_max_atoms,
            "num_elements": 118,
            "cond_fields": cond_fields,
            "cond_normalize_fields": normalize_fields,
            "cond_stats": _serialize_cond_stats(cond_stats),
        },
        "geometry_config": {
            "use_geometry_fields": use_geometry_fields,
            "geom_fields": ["uv_angle", "z_norm", "lattice_param", "t"],
            "use_thickness": use_thickness,
            "t_normalize": t_stats is not None,
            "t_mean": t_stats[0] if t_stats is not None else None,
            "t_std": t_stats[1] if t_stats is not None else None,
            "z_norm_mean": z_norm_stats[0] if z_norm_stats is not None else None,
            "z_norm_std": z_norm_stats[1] if z_norm_stats is not None else None,
        },
        "symmetry_config": {
            "symmetry_mode": symmetry_mode,
            "symmetry_loss_weight": float(symmetry_loss_weight),
            "symmetry_symprec": float(args.symmetry_symprec),
            "wyckoff_constraint": str(args.wyckoff_constraint),
        },
        "lattice_stats": lattice_stats,
        "dataset": {
            "type": "C2DBTokenNPZDataset",
            "npz": str(args.npz),
            "split_json": str(args.split_json) if args.split_json is not None else None,
            "split": str(args.split),
            "pbc_mask": pbc_mask,
            "quality_jsonl": str(args.quality_jsonl) if args.quality_jsonl is not None else None,
            "quality_buckets": _parse_quality_buckets(args.quality_buckets),
            "quality_hard_pass_only": bool(args.quality_hard_pass_only),
            "quality_filter_summary": quality_filter_summary,
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=True)
    metrics_log_path = run_dir / "train_metrics.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    tb_writer = None
    if args.tb_logdir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(args.tb_logdir))
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] TensorBoard not available: {exc}")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=str(args.wandb_entity) if args.wandb_entity else None,
                name=str(args.wandb_name) if args.wandb_name else None,
                dir=str(run_dir),
                config={
                    "model": str(args.model_size),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "vacuum_loss_weight": float(args.vacuum_loss_weight),
                    "cross_vacuum_loss_weight": float(args.cross_vacuum_loss_weight),
                },
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] W&B init failed: {exc}")

    global_step = 0
    best_loss = float("inf")
    if args.drop_last:
        steps_per_epoch = dataset_len // max(1, args.batch_size)
    else:
        steps_per_epoch = math.ceil(dataset_len / max(1, args.batch_size))
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch
    cond_max_schedule = None
    if args.cond_max_start is not None:
        cond_max_schedule = {
            "start": float(args.cond_max_start),
            "end": float(args.cond_max_end) if args.cond_max_end is not None else float(args.cond_max),
            "steps": int(args.cond_max_steps) if args.cond_max_steps is not None else total_steps,
            "schedule": str(args.cond_max_schedule),
        }
    debug_grad_prefixes = [p.strip() for p in args.debug_grad_submodules.split(",") if p.strip()]
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        if args.curriculum_collision and base_indices is not None and min_dist_all is not None:
            epoch_indices = _curriculum_indices(
                base_indices=base_indices,
                min_dist_all=min_dist_all,
                min_dist_cut=float(args.curriculum_min_dist_cut),
                epoch=epoch,
                epochs_ramp=max(1, int(args.curriculum_epochs)),
                seed=args.seed,
            )
            loader = _make_loader(epoch_indices)
        global_step, epoch_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            log_interval=args.log_interval,
            tb_writer=tb_writer,
            tb_interval=args.tb_interval,
            wandb_run=wandb_run,
            alert_steps=int(args.alert_steps),
            alert_collision_rate=float(args.alert_collision_rate),
            alert_min_dist_p10=float(args.alert_min_dist_p10),
            alert_vacuum_gap=float(args.alert_vacuum_gap),
            alert_chol_clamp_rate=float(args.alert_chol_clamp_rate),
            global_step=global_step,
            max_steps=args.max_steps,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            min_lr=args.min_lr,
            schedule=args.lr_schedule,
            clip_grad=args.clip_grad,
            ema=ema,
            use_condition=use_condition,
            use_geometry_fields=use_geometry_fields,
            use_thickness=use_thickness,
            max_atoms=cond_max_atoms,
            num_elements=118,
            cond_stats=cond_stats,
            cond_fields=cond_fields,
            t_stats=t_stats,
            metrics_log_path=metrics_log_path,
            loss_weight_scheduler=loss_weight_scheduler,
            base_loss_weights=base_loss_weights,
            volume_c_len_warmup_steps=int(args.volume_c_len_warmup_steps),
            volume_c_len_warmup_start=float(args.volume_c_len_warmup_start),
            volume_c_len_warmup_end=float(args.volume_c_len_warmup_end),
            cond_max_schedule=cond_max_schedule,
            debug_grad_prefixes=debug_grad_prefixes,
        )
        print(f"[epoch {epoch + 1}] mean loss={epoch_loss:.4f}")

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "mean_loss": epoch_loss,
            "pred_target": "x0",
            "config": model_cfg,
            "diffusion_config": denoiser_cfg.diffusion,
            "optimizer_config": config_payload["optimizer_config"],
            "cond_config": {
                "use_condition": use_condition,
                "cond_dim": cond_dim,
                "max_atoms": cond_max_atoms,
                "num_elements": 118,
                "cond_fields": cond_fields,
                "cond_normalize_fields": normalize_fields,
                "cond_stats": _serialize_cond_stats(cond_stats),
            },
            "geometry_config": {
                "use_geometry_fields": use_geometry_fields,
                "geom_fields": ["uv_angle", "z_norm", "lattice_param", "t"],
                "use_thickness": use_thickness,
                "t_normalize": t_stats is not None,
                "t_mean": t_stats[0] if t_stats is not None else None,
                "t_std": t_stats[1] if t_stats is not None else None,
            },
            "symmetry_config": {
                "symmetry_mode": symmetry_mode,
                "symmetry_loss_weight": float(symmetry_loss_weight),
                "symmetry_symprec": float(args.symmetry_symprec),
                "wyckoff_constraint": str(args.wyckoff_constraint),
            },
        }

        last_path = run_dir / "atomdenoiser_last.pt"
        save_checkpoint(last_path, ckpt_payload)
        print(f"Saved latest checkpoint to {last_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = run_dir / "atomdenoiser_best.pt"
            save_checkpoint(best_path, {**ckpt_payload, "best_loss": best_loss})
            print(f"Updated best checkpoint to {best_path} (mean loss {best_loss:.4f})")

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break

    metrics_aggregate = _aggregate_train_metrics_jsonl(metrics_log_path)
    metrics_aggregate_path = run_dir / "train_metrics_aggregate.json"
    with metrics_aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_aggregate, f, indent=2, ensure_ascii=True)
    print(f"[info] wrote train metrics aggregate to {metrics_aggregate_path}")

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
