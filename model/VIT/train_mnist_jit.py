from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Make the upstream JiT implementation importable.
JIT_DIR = Path(__file__).resolve().parent / "JiT"
if str(JIT_DIR) not in sys.path:
    sys.path.append(str(JIT_DIR))

from denoiser import Denoiser  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MNIST JiT reproduction")
    parser.add_argument("--data-dir", type=str, default="data/mnist", help="Where to download/cache MNIST")
    parser.add_argument("--output-dir", type=str, default="outputs/jit_mnist", help="Checkpoint + sample output folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (defaults to auto-detect)")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from")
    parser.add_argument("--log-every", type=int, default=50, help="Iterations between loss prints")
    parser.add_argument("--sample-every", type=int, default=2, help="Epoch interval for saving sample grids")

    # JiT-specific knobs (tuned for MNIST)
    parser.add_argument("--model", type=str, default="JiT-T/4-MNIST")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--proj-dropout", type=float, default=0.0)
    parser.add_argument("--P-mean", dest="P_mean", type=float, default=-0.3)
    parser.add_argument("--P-std", dest="P_std", type=float, default=0.7)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--t-eps", dest="t_eps", type=float, default=5e-3)
    parser.add_argument("--label-drop-prob", type=float, default=0.2)
    parser.add_argument("--ema-decay1", type=float, default=0.999)
    parser.add_argument("--ema-decay2", type=float, default=0.995)
    parser.add_argument("--sampling-method", type=str, default="heun")
    parser.add_argument("--num-sampling-steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--interval-min", type=float, default=0.0)
    parser.add_argument("--interval-max", type=float, default=1.0)
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
        ]
    )
    dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def move_ema_to_device(params: Iterable[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [p.to(device) for p in params]


def load_checkpoint(
    model: Denoiser, optimizer: optim.Optimizer, ckpt_path: Path, device: torch.device
) -> int:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.ema_params1 = move_ema_to_device(checkpoint["ema1"], device)
    model.ema_params2 = move_ema_to_device(checkpoint["ema2"], device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)) + 1


def save_checkpoint(model: Denoiser, optimizer: optim.Optimizer, epoch: int, ckpt_path: Path) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "ema1": [p.detach().cpu() for p in model.ema_params1],
        "ema2": [p.detach().cpu() for p in model.ema_params2],
    }
    torch.save(ckpt, ckpt_path)


def apply_ema_weights(model: Denoiser, ema_params: List[torch.Tensor]) -> List[torch.Tensor]:
    """Swap model parameters with EMA weights; return the originals for restoration."""
    originals: List[torch.Tensor] = []
    for param, ema in zip(model.parameters(), ema_params):
        originals.append(param.detach().clone())
        param.data.copy_(ema)
    return originals


def restore_weights(model: Denoiser, originals: List[torch.Tensor]) -> None:
    for param, orig in zip(model.parameters(), originals):
        param.data.copy_(orig)


@torch.no_grad()
def sample_and_save(model: Denoiser, device: torch.device, args: argparse.Namespace, epoch: int) -> None:
    model.eval()
    labels = torch.arange(args.num_classes, device=device)
    swapped = apply_ema_weights(model, model.ema_params1)
    samples = model.generate(labels)
    restore_weights(model, swapped)

    grid = utils.make_grid((samples + 1) / 2.0, nrow=5, padding=2, normalize=False)
    out_path = Path(args.output_dir) / f"samples_epoch{epoch}.png"
    utils.save_image(grid, out_path)
    print(f"[epoch {epoch}] saved samples -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.device:
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = requested
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Align argument names expected by the upstream JiT components.
    args.class_num = args.num_classes

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    dataloader = build_dataloader(args, device)
    model = Denoiser(args).to(device)
    model.ema_params1 = [p.detach().clone() for p in model.parameters()]
    model.ema_params2 = [p.detach().clone() for p in model.parameters()]

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    ckpt_path = Path(args.resume) if args.resume else Path(args.output_dir) / "checkpoint.pt"
    if ckpt_path.exists():
        start_epoch = load_checkpoint(model, optimizer, ckpt_path, device)
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Using device={device}, model={args.model}, img_size={args.img_size}, lr={args.learning_rate}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_enabled else contextlib.nullcontext()
            with ctx:
                loss = model(images, labels)
            loss.backward()
            optimizer.step()
            model.update_ema()

            running_loss += loss.item()
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                print(f"[epoch {epoch} step {step}] loss={avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, ckpt_path)
        if (epoch + 1) % args.sample_every == 0:
            sample_and_save(model, device, args, epoch)

    # Final sample for convenience
    sample_and_save(model, device, args, args.epochs)


if __name__ == "__main__":
    main()
