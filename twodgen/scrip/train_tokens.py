from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler

from twodgen.data.c2db_dataset import C2DBAtomDataset, C2DBTokenNPZDataset
from twodgen.data.splits import load_c2db_split, select_split_indices, validate_split_indices
from twodgen.common.crystal import gram6_to_lattice, frac_mic_dist
from twodgen.common.run_metadata import collect_run_metadata
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.model.atom_transformer import AtomTransformerConfig
from twodgen.model.model_sizes import resolve_model_hparams


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
    if isinstance(dataset, C2DBTokenNPZDataset):
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

        lr = _adjust_lr(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr, schedule)
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
                min_dist_mean = float(np.mean(min_dist_batch)) if min_dist_batch.size else float("nan")
                min_dist_p10 = float(np.percentile(min_dist_batch, 10.0)) if min_dist_batch.size else float("nan")
                collision_rate = float(np.mean(min_dist_batch < model.cfg.min_dist_train_cut)) if min_dist_batch.size else 0.0
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
            msg += f" min_dist_mean={min_dist_mean:.3f} min_dist_p10={min_dist_p10:.3f} collision_rate={collision_rate:.3f}"
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
            print(msg)
            if metrics_log_path is not None:
                payload = {
                    "step": global_step,
                    "loss": float(loss.item()),
                    "loss_f": float(metrics["loss_f"].item()),
                    "loss_g": float(metrics["loss_g"].item()),
                    "loss_z": float(metrics["loss_z"].item()),
                    "lr": float(lr),
                    "min_dist_mean": min_dist_mean,
                    "min_dist_p10": min_dist_p10,
                    "collision_rate": collision_rate,
                }
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

    use_condition = args.use_condition
    geom_available = all(
        getattr(dataset, name, None) is not None for name in ("uv_angle", "z_norm", "lattice_param")
    )
    if (
        args.use_geometry_fields
        and isinstance(dataset, C2DBTokenNPZDataset)
        and getattr(dataset, "order_idx", None) is not None
        and not dataset.align_atoms
    ):
        raise ValueError(
            "Geometry fields enabled but align_atoms is False while order_idx exists. "
            "Set --align-atoms or disable geometry heads."
        )
    t_available = getattr(dataset, "t", None) is not None
    use_geometry_fields = args.use_geometry_fields and geom_available
    if args.use_geometry_fields and not geom_available:
        print("[warn] geometry fields not found in dataset; disabling geometry heads.")
    use_thickness = use_geometry_fields and t_available
    if use_geometry_fields and not t_available:
        print("[warn] thickness field not found in dataset; disabling t head.")
    t_stats = None
    if use_thickness:
        t_values = getattr(dataset, "t", None)
        if t_values is not None:
            if getattr(dataset, "cond_t_mean", None) is not None and getattr(dataset, "cond_t_std", None) is not None:
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
    cond_max_atoms = int(getattr(dataset, "max_atoms", args.max_atoms))
    cond_dim = 0
    cond_stats = {}
    cond_fields = _resolve_cond_fields(args)
    normalize_fields = _parse_cond_fields(args.cond_normalize_fields)
    element_ids = _parse_element_ids(args.element_ids)
    if use_condition:
        sample = dataset[split_indices[0]] if split_indices is not None else dataset[0]
        cond_dim = _infer_cond_dim(sample, cond_fields, num_elements=118)
        cond_stats = _compute_cond_stats(dataset, cond_fields, normalize_fields, indices=split_indices)

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

    if args.dual_graph and args.edge_type_dim == 0:
        args.edge_type_dim = 4
        print("[warn] --dual-graph enabled with edge_type_dim=0; defaulting to 4.")

    dataset_len = len(split_indices) if split_indices is not None else len(dataset)
    if not args.drop_last and dataset_len < args.batch_size:
        print("[warn] --no-drop-last with dataset smaller than batch size; expect tiny batches.")

    loader = prepare_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        args.bucket_batches,
        args.bucket_shuffle,
        seed=args.seed,
        drop_last=args.drop_last,
        indices=split_indices,
    )

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
    denoiser_cfg.diffusion.cell_init = args.cell_init
    denoiser_cfg.diffusion.cell_init_scale = args.cell_init_scale
    denoiser_cfg.diffusion.cell_init_noise = args.cell_init_noise
    if use_condition:
        denoiser_cfg.diffusion.cond_drop_prob = args.cond_drop_prob
    denoiser_cfg.diffusion.use_uncertainty_weighting = not args.no_uncertainty_weighting
    model = AtomDenoiser(denoiser_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.3f}M (model_size={args.model_size})")
    print("[info] pred_target=x0")
    betas = _parse_betas(args.betas)
    param_groups = _build_param_groups(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=betas)
    ema = EMA(model, args.ema_decay) if args.ema else None

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
        "dataset": {
            "type": "C2DBTokenNPZDataset" if args.npz is not None else "C2DBAtomDataset",
            "npz": str(args.npz) if args.npz is not None else None,
            "csv": str(args.csv) if args.csv is not None else None,
            "split_json": str(args.split_json) if args.split_json is not None else None,
            "split": str(args.split),
            "pbc_mask": pbc_mask,
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=True)
    metrics_log_path = run_dir / "train_metrics.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    global_step = 0
    best_loss = float("inf")
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * len(loader)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
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
