"""
Legacy training loop for torus-encoded 3x24x24 grids using C2DBDenoiser.

Usage:
    uv run python 2DGEN/scrip/train.py \
        --data data/C2DB/ache/c2db_grid.npz \
        --epochs 1 --batch-size 64 --lr 1e-4

Only keep two checkpoints: latest (last epoch) and best (lowest mean loss).
Token-based diffusion is the default path; see 2DGEN/scrip/train_tokens.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Ensure the 2DGEN package directory is importable even though the folder starts with a digit.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.c2db_dataset import C2DBGridNPZDataset  # noqa: E402
from data.torus import torus_feature_dim  # noqa: E402
from model.denoiser import C2DBDenoiser, DenoiserConfig  # noqa: E402
from model.model import JiTC2DBConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C2DBDenoiser on preprocessed torus-encoded grids.")
    parser.add_argument("--data", type=Path, default=Path("data/C2DB/ache/c2db_grid.npz"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on total training steps.")
    return parser.parse_args()


def prepare_dataloader(dataset: C2DBGridNPZDataset, batch_size: int, num_workers: int) -> DataLoader:
    def collate(batch):
        # Batch may contain (sample,) or (sample, material_id)
        if isinstance(batch[0], Tuple) or isinstance(batch[0], list):
            xs = [b[0] for b in batch]
        else:
            xs = batch
        x_tensor = torch.stack(xs, dim=0)
        return x_tensor

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    return loader


def train_one_epoch(
    model: nn.Module,
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
    for step, batch in enumerate(loader):
        x0 = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(x0)
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = C2DBGridNPZDataset(args.data)
    if dataset.torus_freqs is not None:
        print(f"Torus freqs from dataset: {dataset.torus_freqs}")

    loader = prepare_dataloader(dataset, args.batch_size, args.num_workers)

    _, in_chans, height, width = dataset.x.shape
    expected_width = None
    if dataset.torus_freqs is not None:
        expected_width = torus_feature_dim(dataset.torus_freqs)
    if expected_width is not None and width != expected_width:
        print(f"[warn] dataset width {width} != torus_feature_dim({dataset.torus_freqs})={expected_width}")

    model_cfg = JiTC2DBConfig(img_size=(height, width), in_chans=in_chans)
    model = C2DBDenoiser(DenoiserConfig(model=model_cfg)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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
        }

        # Save latest checkpoint (overwrite)
        last_path = args.save_dir / "c2dbdenoiser_last.pt"
        save_checkpoint(last_path, ckpt_payload)
        print(f"Saved latest checkpoint to {last_path}")

        # Save best checkpoint (by lowest mean loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = args.save_dir / "c2dbdenoiser_best.pt"
            save_checkpoint(
                best_path,
                {
                    **ckpt_payload,
                    "best_loss": best_loss,
                },
            )
            print(f"Updated best checkpoint to {best_path} (mean loss {best_loss:.4f})")

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break


if __name__ == "__main__":
    main()
