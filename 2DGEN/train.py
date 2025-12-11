"""
Simple training loop for C2DB 3x24x3 grids using C2DBDenoiser.

Usage:
    uv run python 2DGEN/train.py \
        --data data/C2DB/ache/c2db_grid.npz \
        --epochs 1 --batch-size 64 --lr 1e-4
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
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from data.c2db_dataset import C2DBGridNPZDataset  # noqa: E402
from model.denoiser import C2DBDenoiser  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C2DBDenoiser on preprocessed 3x24x3 grids.")
    parser.add_argument("--data", type=Path, default=Path("data/C2DB/ache/c2db_grid.npz"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on total training steps.")
    return parser.parse_args()


def prepare_dataloader(data_path: Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = C2DBGridNPZDataset(data_path)

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
) -> int:
    model.train()
    for step, batch in enumerate(loader):
        x0 = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(x0)
        loss.backward()
        optimizer.step()

        if global_step % log_interval == 0:
            print(f"[step {global_step}] loss={loss.item():.4f}")

        global_step += 1
        if max_steps is not None and global_step >= max_steps:
            break
    return global_step


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = prepare_dataloader(args.data, args.batch_size, args.num_workers)
    model = C2DBDenoiser().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        global_step = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            log_interval=args.log_interval,
            global_step=global_step,
            max_steps=args.max_steps,
        )
        # Save checkpoint each epoch
        ckpt_path = args.save_dir / f"c2dbdenoiser_epoch{epoch+1}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break


if __name__ == "__main__":
    main()
