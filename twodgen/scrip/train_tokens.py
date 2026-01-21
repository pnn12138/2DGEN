from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler

from twodgen.data.c2db_dataset import C2DBAtomDataset, C2DBTokenNPZDataset
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
    cfg.lambda_cond = float(weights.get("cond", cfg.lambda_cond))
    cfg.lambda_chol_bound = float(weights.get("chol_bound", cfg.lambda_chol_bound))
    cfg.lambda_expand_collision = float(weights.get("expand_collision", cfg.lambda_expand_collision))
    cfg.lambda_volume = float(weights.get("volume", cfg.lambda_volume))
    cfg.lambda_c_len = float(weights.get("c_len", cfg.lambda_c_len))


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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl"],
        help="Model size preset.",
    )
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
        help="Number of epochs to linearly ramp collision-risk samples from 0% to 100%.",
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
        default=0.1,
        help="Weight for 2D vacuum loss (0 disables).",
    )
    parser.add_argument(
        "--vacuum-min",
        type=float,
        default=15.0,
        help="Minimum vacuum thickness (Angstrom) for vacuum loss.",
    )
    parser.add_argument(
        "--vacuum-loss-power",
        type=int,
        default=2,
        help="Power for vacuum loss: 1=linear, 2=squared, ...",
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
        "--c-len-loss-weight",
        type=float,
        default=0.02,
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
        default=0.02,
        help="Weight for lattice volume penalty during training.",
    )
    parser.add_argument(
        "--volume-min",
        type=float,
        default=1.0,
        help="Minimum lattice volume enforced by the volume penalty.",
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
        default=30000,
        help="Warmup steps for selected loss weights (0 disables).",
    )
    parser.add_argument(
        "--loss-weight-warmup-keys",
        type=str,
        default="vacuum,cond,chol_bound,expand_collision,volume,c_len",
        help="Comma-separated loss keys for warmup (vacuum,cond,chol_bound,expand_collision).",
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
        default="linear",
        help="Schedule shape for warmup (linear/sigmoid/cosine).",
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
        csv=None,
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
        niggli_reduce=False,
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
    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
        if dataset.material_ids is None:
            return None
        return dataset.material_ids[idx]
    if isinstance(dataset, C2DBAtomDataset):
        try:
            return dataset.base.get_metadata(idx).material_id
        except IndexError:
            return None
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


def _atom_counts(dataset: C2DBAtomDataset | C2DBTokenNPZDataset) -> torch.Tensor:
    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
        return dataset.atom_mask.sum(dim=1)
    counts = []
    for i in range(len(dataset)):
        counts.append(dataset[i]["atom_mask"].sum())
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
    dataset: C2DBAtomDataset | C2DBTokenNPZDataset,
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
    dataset: C2DBAtomDataset | C2DBTokenNPZDataset,
    *,
    pbc_mask: tuple[int, int, int],
    g_scale: float,
    batch_size: int = 256,
) -> torch.Tensor:
    dataset_for_iter = _unwrap_indexed_dataset(dataset)

    if isinstance(dataset_for_iter, C2DBTokenNPZDataset):
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


def _estimate_scube_stats(
    dataset: C2DBAtomDataset | C2DBTokenNPZDataset, g_scale: float, indices: list[int] | None = None
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
        if field not in sample:
            raise ValueError(f"{field} not found but requested by --cond-fields.")
        value = sample[field]
        cond_dim += int(value.numel())
    return cond_dim


def _compute_cond_stats(
    dataset: C2DBAtomDataset | C2DBTokenNPZDataset,
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
) -> tuple[int, float]:
    model.train()
    total_loss = 0.0
    step_count = 0
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

        lr = _adjust_lr(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr, schedule)
        loss_weight_state = None
        if loss_weight_scheduler is not None:
            loss_weight_state = loss_weight_scheduler.weights(global_step)
            _apply_loss_weights(model, loss_weight_state)
        optimizer.zero_grad(set_to_none=True)
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
        )
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
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
                if model.cfg.model.cell_rep == "cholesky6" and (
                    model.cfg.model.chol_log_min is not None
                    or model.cfg.model.chol_log_max is not None
                ):
                    diag = gram6_to_cholesky6(gram6, log_min=None, log_max=None)[:, :3]
                    diag_np = diag.detach().cpu().numpy()
                    hit = np.zeros((diag_np.shape[0],), dtype=bool)
                    if model.cfg.model.chol_log_min is not None:
                        hit |= diag_np.min(axis=1) <= float(model.cfg.model.chol_log_min) + 1e-4
                    if model.cfg.model.chol_log_max is not None:
                        hit |= diag_np.max(axis=1) >= float(model.cfg.model.chol_log_max) - 1e-4
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
            if "loss_angle" in metrics:
                msg += f" loss_angle={metrics['loss_angle'].item():.4f}"
            if "loss_cond" in metrics:
                msg += f" loss_cond={metrics['loss_cond'].item():.4f}"
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
            msg += (
                f" min_dist_mean={min_dist_mean:.3f} min_dist_p10={min_dist_p10:.3f}"
                f" collision_rate={collision_rate:.3f} min_dist_inf={min_dist_inf}"
                f" low_atoms={min_dist_low_atoms}"
            )
            if chol_log_clamp_rate is not None:
                msg += f" chol_log_clamp={chol_log_clamp_rate:.3f}"
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
            print(msg)
            if metrics_log_path is not None:
                payload = {
                    "step": global_step,
                    "loss": float(loss.item()),
                    "loss_f": float(metrics["loss_f"].item()),
                    "loss_g": float(metrics["loss_g"].item()),
                    "loss_z": float(metrics["loss_z"].item()),
                    "loss_comp": float(metrics.get("loss_comp", torch.tensor(0.0)).item()),
                    "loss_vacuum": float(metrics.get("loss_vacuum", torch.tensor(0.0)).item()),
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
                if "loss_cond" in metrics:
                    payload["loss_cond"] = float(metrics["loss_cond"].item())
                if "loss_volume" in metrics:
                    payload["loss_volume"] = float(metrics["loss_volume"].item())
                if "loss_c_len" in metrics:
                    payload["loss_c_len"] = float(metrics["loss_c_len"].item())
                if "loss_chol_bound" in metrics:
                    payload["loss_chol_bound"] = float(metrics["loss_chol_bound"].item())
                if "loss_expand_collision" in metrics:
                    payload["loss_expand_collision"] = float(metrics["loss_expand_collision"].item())
                if "pred_angle_out_rate" in metrics:
                    payload["pred_angle_out_rate"] = float(metrics["pred_angle_out_rate"].item())
                if "pred_cond_mean" in metrics:
                    payload["pred_cond_mean"] = float(metrics["pred_cond_mean"].item())
                if "chol_bound_rate" in metrics:
                    payload["chol_bound_rate"] = float(metrics["chol_bound_rate"].item())
                if "min_dist_pred_mean" in metrics:
                    payload["min_dist_pred_mean"] = float(metrics["min_dist_pred_mean"].item())
                if "min_dist_pred_p10" in metrics:
                    payload["min_dist_pred_p10"] = float(metrics["min_dist_pred_p10"].item())
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
                if "s_cond" in metrics:
                    payload["s_cond"] = float(metrics["s_cond"].item())
                if loss_weight_state is not None:
                    payload["lambda_vacuum"] = float(loss_weight_state.get("vacuum", 0.0))
                    payload["lambda_cond"] = float(loss_weight_state.get("cond", 0.0))
                    payload["lambda_chol_bound"] = float(loss_weight_state.get("chol_bound", 0.0))
                    payload["lambda_expand_collision"] = float(
                        loss_weight_state.get("expand_collision", 0.0)
                    )
                    payload["lambda_volume"] = float(loss_weight_state.get("volume", 0.0))
                    payload["lambda_c_len"] = float(loss_weight_state.get("c_len", 0.0))
                with metrics_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")

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

    split_indices: list[int] | None = None
    if args.npz is not None:
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
    else:
        if args.split_json is not None and args.split != "all":
            raise ValueError("--split-json/--split are supported only when training from --npz.")
        csv_path = args.csv if args.csv is not None else Path("data/C2DB/c2db_summary.csv")
        dataset = C2DBAtomDataset(
            csv_path,
            max_atoms=args.max_atoms,
            g_scale=args.g_scale,
            niggli_reduce=args.niggli_reduce,
        )
        g_scale = args.g_scale
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
            use_dataset_stats = (
                split_indices is None
                and getattr(dataset, "cond_t_mean", None) is not None
                and getattr(dataset, "cond_t_std", None) is not None
            )
            if use_dataset_stats:
                t_mean = float(dataset.cond_t_mean.reshape(-1)[0])
                t_std = float(dataset.cond_t_std.reshape(-1)[0])
            else:
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
    )
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    denoiser_cfg.diffusion.mode = args.mode
    denoiser_cfg.diffusion.cell_rep = args.cell_rep
    denoiser_cfg.diffusion.chol_log_min = args.chol_log_min
    denoiser_cfg.diffusion.chol_log_max = args.chol_log_max
    denoiser_cfg.diffusion.lambda_comp = float(args.comp_loss_weight)
    denoiser_cfg.diffusion.comp_loss_mode = str(args.comp_loss_mode)
    denoiser_cfg.diffusion.lambda_vacuum = float(args.vacuum_loss_weight)
    denoiser_cfg.diffusion.vacuum_min = float(args.vacuum_min)
    denoiser_cfg.diffusion.vacuum_loss_power = int(args.vacuum_loss_power)
    denoiser_cfg.diffusion.lambda_angle = float(args.angle_loss_weight)
    denoiser_cfg.diffusion.angle_min = float(args.angle_min)
    denoiser_cfg.diffusion.angle_max = float(args.angle_max)
    denoiser_cfg.diffusion.lambda_cond = float(args.cond_loss_weight)
    denoiser_cfg.diffusion.cond_max = float(args.cond_max)
    denoiser_cfg.diffusion.lambda_volume = float(args.volume_loss_weight)
    denoiser_cfg.diffusion.volume_min = float(args.volume_min)
    denoiser_cfg.diffusion.lambda_c_len = float(args.c_len_loss_weight)
    denoiser_cfg.diffusion.c_len_min = float(args.c_len_min)
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
    warmup_keys = _parse_loss_weight_keys(args.loss_weight_warmup_keys)
    if args.loss_weight_warmup_steps > 0 and warmup_keys:
        base_weights = {
            "vacuum": float(args.vacuum_loss_weight),
            "cond": float(args.cond_loss_weight),
            "chol_bound": float(args.chol_bound_loss_weight),
            "expand_collision": float(args.expand_on_collision_weight),
            "volume": float(args.volume_loss_weight),
            "c_len": float(args.c_len_loss_weight),
        }
        schedule_cfg = LossWeightScheduleConfig(
            warmup_steps=int(args.loss_weight_warmup_steps),
            start_factor=float(args.loss_weight_warmup_start),
            end_factor=float(args.loss_weight_warmup_end),
            keys=tuple(warmup_keys),
            schedule=str(args.loss_weight_schedule),
        )
        loss_weight_scheduler = LossWeightScheduler(base_weights, schedule_cfg)

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
        "lattice_stats": lattice_stats,
        "dataset": {
            "type": "C2DBTokenNPZDataset" if args.npz is not None else "C2DBAtomDataset",
            "npz": str(args.npz) if args.npz is not None else None,
            "csv": str(args.csv) if args.csv is not None else None,
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

    global_step = 0
    best_loss = float("inf")
    if args.drop_last:
        steps_per_epoch = dataset_len // max(1, args.batch_size)
    else:
        steps_per_epoch = math.ceil(dataset_len / max(1, args.batch_size))
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch
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


if __name__ == "__main__":
    main()
