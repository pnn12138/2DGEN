from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler

from twodgen.data.c2db_dataset import C2DBAtomDataset, C2DBTokenNPZDataset
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.model.atom_transformer import AtomTransformerConfig
from twodgen.model.model_sizes import resolve_model_hparams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train token-based crystal diffusion model.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic algorithms (may be slower).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl"],
        help="Model size preset controlling Transformer width/depth (overridable by explicit --*-dim/--depth flags).",
    )
    parser.add_argument("--embed-dim", type=int, default=None, help="Transformer embedding dim (override preset).")
    parser.add_argument("--depth", type=int, default=None, help="Transformer depth (override preset).")
    parser.add_argument("--num-heads", type=int, default=None, help="Attention heads (override preset).")
    parser.add_argument("--mlp-ratio", type=float, default=None, help="MLP expansion ratio (override preset).")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (override preset).")
    parser.add_argument("--time-embed-dim", type=int, default=None, help="Timestep embedding dim (override preset).")
    parser.add_argument("--z-embed-dim", type=int, default=None, help="Element token embedding dim (override preset).")
    parser.add_argument("--f-embed-dim", type=int, default=None, help="Frac token embedding dim (override preset).")
    parser.add_argument("--rbf-dim", type=int, default=None, help="RBF distance embedding dim (override preset).")
    parser.add_argument("--pair-mlp-hidden", type=int, default=None, help="Pair MLP hidden dim (override preset).")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None, help="Preprocessed token cache (npz).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--betas", type=str, default="0.9,0.95")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--g-scale", type=float, default=100.0)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mode", type=str, default="diffusion", choices=["diffusion", "flow"])
    parser.add_argument("--no-uncertainty-weighting", action="store_true")
    parser.add_argument("--cell-rep", type=str, default="gram6", choices=["gram6", "cholesky6"])
    parser.add_argument("--chol-log-min", type=float, default=None)
    parser.add_argument("--chol-log-max", type=float, default=None)
    parser.add_argument("--cell-init", type=str, default="gaussian", choices=["gaussian", "iso"])
    parser.add_argument("--cell-init-scale", type=float, default=None)
    parser.add_argument("--cell-init-noise", type=float, default=None)
    parser.add_argument("--cell-init-scale-factor", type=float, default=1.5)
    parser.add_argument("--cell-log-min-factor", type=float, default=0.7)
    parser.add_argument("--cell-log-max-factor", type=float, default=2.5)
    parser.add_argument("--niggli-reduce", action="store_true", help="Apply Niggli reduction on-the-fly (CSV).")
    parser.add_argument("--bucket-batches", action="store_true", help="Bucket batches by atom count to reduce padding.")
    parser.add_argument("--bucket-shuffle", action="store_true", help="Shuffle within/among buckets.")
    parser.add_argument("--use-condition", action="store_true", help="Condition on counts/lattice parameters.")
    parser.add_argument("--cond-drop-prob", type=float, default=0.1, help="Condition dropout prob for CFG-style training.")
    parser.add_argument(
        "--cond-fields",
        type=str,
        default=None,
        help="Comma-separated condition fields (e.g. counts_vector,lattice_param,t,xrd).",
    )
    parser.add_argument(
        "--cond-normalize-fields",
        type=str,
        default="",
        help="Comma-separated condition fields to z-score normalize.",
    )
    parser.add_argument(
        "--pbc-mask",
        type=str,
        default="1,1,0",
        help="Comma-separated PBC mask for MIC distance, e.g. 1,1,0 for slab.",
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
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id + 1
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if use_buckets:
        counts = _atom_counts(dataset).float()
        sampler = BucketBatchSampler(counts, batch_size=batch_size, shuffle=shuffle, generator=generator)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        generator=generator,
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
    dataset: C2DBAtomDataset | C2DBTokenNPZDataset, g_scale: float
) -> tuple[float, float, float, float]:
    if isinstance(dataset, C2DBTokenNPZDataset):
        gram6 = dataset.gram6.float()
    else:
        gram6_list = []
        for i in range(len(dataset)):
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
) -> dict:
    stats: dict[str, torch.Tensor] = {}
    if not isinstance(dataset, C2DBTokenNPZDataset):
        return stats
    for field in normalize_fields:
        if field in ("counts", "counts_vector"):
            continue
        if field not in cond_fields:
            continue
        value = getattr(dataset, field, None)
        if value is None:
            continue
        value = value.float()
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
    max_atoms: int,
    num_elements: int,
    cond_stats: dict | None = None,
    cond_fields: Optional[list[str]] = None,
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
        cond = None
        if use_condition:
            cond = _build_condition(
                batch,
                max_atoms=max_atoms,
                num_elements=num_elements,
                cond_stats=cond_stats,
                cond_fields=cond_fields,
            ).to(device, non_blocking=True)

        lr = _adjust_lr(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr, schedule)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, metrics = model(z, frac, atom_mask, gram6, cond)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        if global_step % log_interval == 0:
            msg = (
                f"[step {global_step}] loss={loss.item():.4f} "
                f"loss_f={metrics['loss_f'].item():.4f} "
                f"loss_g={metrics['loss_g'].item():.4f} "
                f"loss_z={metrics['loss_z'].item():.4f} "
                f"lr={lr:.6e}"
            )
            if "s_f" in metrics:
                msg += (
                    f" s_f={metrics['s_f'].item():.3f}"
                    f" s_g={metrics['s_g'].item():.3f}"
                    f" s_z={metrics['s_z'].item():.3f}"
                )
            print(msg)
            if metrics_log_path is not None:
                payload = {
                    "step": global_step,
                    "loss": float(loss.item()),
                    "loss_f": float(metrics["loss_f"].item()),
                    "loss_g": float(metrics["loss_g"].item()),
                    "loss_z": float(metrics["loss_z"].item()),
                    "lr": float(lr),
                }
                if "s_f" in metrics:
                    payload.update(
                        {
                            "s_f": float(metrics["s_f"].item()),
                            "s_g": float(metrics["s_g"].item()),
                            "s_z": float(metrics["s_z"].item()),
                        }
                    )
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

    if args.npz is not None:
        dataset = C2DBTokenNPZDataset(args.npz)
        if args.g_scale != dataset.g_scale:
            print(f"[warn] g_scale {args.g_scale} != dataset g_scale {dataset.g_scale}")
        g_scale = dataset.g_scale
    else:
        csv_path = args.csv if args.csv is not None else Path("data/C2DB/c2db_summary.csv")
        dataset = C2DBAtomDataset(
            csv_path,
            max_atoms=args.max_atoms,
            g_scale=args.g_scale,
            niggli_reduce=args.niggli_reduce,
        )
        g_scale = args.g_scale

    use_condition = args.use_condition
    cond_max_atoms = int(getattr(dataset, "max_atoms", args.max_atoms))
    cond_dim = 0
    cond_stats = {}
    cond_fields = _resolve_cond_fields(args)
    normalize_fields = _parse_cond_fields(args.cond_normalize_fields)
    if use_condition:
        sample = dataset[0]
        cond_dim = _infer_cond_dim(sample, cond_fields, num_elements=118)
        cond_stats = _compute_cond_stats(dataset, cond_fields, normalize_fields)

    if args.cell_rep == "cholesky6":
        s10, s50, s90, log_std = _estimate_scube_stats(dataset, g_scale)
        if args.cell_init == "iso" and args.cell_init_scale is None:
            args.cell_init_scale = args.cell_init_scale_factor * s50
        if args.cell_init_noise is None:
            args.cell_init_noise = float(min(max(log_std, 0.1), 0.2))
        if args.chol_log_min is None:
            args.chol_log_min = float(np.log(max(args.cell_log_min_factor * s10, 1e-6)))
        if args.chol_log_max is None:
            args.chol_log_max = float(np.log(max(args.cell_log_max_factor * s90, 1e-6)))

    loader = prepare_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        args.bucket_batches,
        args.bucket_shuffle,
        seed=args.seed,
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
        pbc_mask=pbc_mask,
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
    betas = _parse_betas(args.betas)
    param_groups = _build_param_groups(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=betas)
    ema = EMA(model, args.ema_decay) if args.ema else None

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "created_at": run_stamp,
        "args": _serialize_args(args),
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
            max_atoms=cond_max_atoms,
            num_elements=118,
            cond_stats=cond_stats,
            cond_fields=cond_fields,
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
