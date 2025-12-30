from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Sampler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.c2db_dataset import C2DBAtomDataset, C2DBTokenNPZDataset  # noqa: E402
from model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig  # noqa: E402
from model.atom_transformer import AtomTransformerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train token-based crystal diffusion model.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None, help="Preprocessed token cache (npz).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    return parser.parse_args()


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, counts: torch.Tensor, batch_size: int, shuffle: bool) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        order = torch.argsort(counts)
        self.buckets = [
            order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)
        ]
        if self.shuffle:
            perm = torch.randperm(len(self.buckets)).tolist()
            self.buckets = [self.buckets[i] for i in perm]
            for bucket in self.buckets:
                perm_in = torch.randperm(len(bucket)).tolist()
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


def prepare_dataloader(dataset: C2DBAtomDataset | C2DBTokenNPZDataset, batch_size: int, num_workers: int, use_buckets: bool, shuffle: bool) -> DataLoader:
    if use_buckets:
        counts = _atom_counts(dataset).float()
        sampler = BucketBatchSampler(counts, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )


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
    return det_g.pow(1.0 / 6.0) * (g_scale ** 0.5)


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


def train_one_epoch(
    model: AtomDenoiser,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    global_step: int,
    max_steps: int | None,
) -> tuple[int, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        z = batch["atomic_numbers"].to(device, non_blocking=True)
        frac = batch["frac_coords"].to(device, non_blocking=True)
        atom_mask = batch["atom_mask"].to(device, non_blocking=True)
        gram6 = batch["gram6"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = model(z, frac, atom_mask, gram6)
        loss.backward()
        optimizer.step()

        if global_step % log_interval == 0:
            print(f"[step {global_step}] loss={loss.item():.4f}")

        global_step += 1
        total_loss += loss.item()
        total_steps += 1
        if max_steps is not None and global_step >= max_steps:
            break

    mean_loss = total_loss / max(total_steps, 1)
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


def main() -> None:
    args = parse_args()
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

    loader = prepare_dataloader(dataset, args.batch_size, args.num_workers, args.bucket_batches, args.bucket_shuffle)

    model_cfg = AtomTransformerConfig(
        num_elements=118,
        k_neighbors=args.k_neighbors,
        g_scale=g_scale,
        cell_rep=args.cell_rep,
        chol_log_min=args.chol_log_min,
        chol_log_max=args.chol_log_max,
    )
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    denoiser_cfg.diffusion.mode = args.mode
    denoiser_cfg.diffusion.cell_rep = args.cell_rep
    denoiser_cfg.diffusion.chol_log_min = args.chol_log_min
    denoiser_cfg.diffusion.chol_log_max = args.chol_log_max
    denoiser_cfg.diffusion.cell_init = args.cell_init
    denoiser_cfg.diffusion.cell_init_scale = args.cell_init_scale
    denoiser_cfg.diffusion.cell_init_noise = args.cell_init_noise
    denoiser_cfg.diffusion.use_uncertainty_weighting = not args.no_uncertainty_weighting
    model = AtomDenoiser(denoiser_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "created_at": run_stamp,
        "args": _serialize_args(args),
        "model_config": asdict(model_cfg),
        "diffusion_config": asdict(denoiser_cfg.diffusion),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=True)

    global_step = 0
    best_loss = float("inf")
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
        )
        print(f"[epoch {epoch + 1}] mean loss={epoch_loss:.4f}")

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "mean_loss": epoch_loss,
            "config": model_cfg,
            "diffusion_config": denoiser_cfg.diffusion,
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
