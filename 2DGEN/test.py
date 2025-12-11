"""
Quick smoke test for C2DBJiT + C2DBDenoiser using velocity loss.

Runs a tiny forward/backward on random 3x24x3 tensors to validate shapes and loss plumbing.
Execute with project env: `uv run python 2DGEN/test.py`
"""

import sys
from pathlib import Path

import torch

# Ensure the 2DGEN package directory is importable even though the folder starts with a digit.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    from model.denoiser import C2DBDenoiser

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = C2DBDenoiser().to(device)

    x0 = torch.randn(4, 3, 24, 3, device=device)
    loss, x_pred, t = model(x0)

    loss.backward()  # ensure grads flow

    print(f"loss: {loss.item():.4f}")
    print(f"x_pred shape: {tuple(x_pred.shape)}")
    print(f"t shape: {tuple(t.shape)}")

    # quick sampler smoke test (few steps to keep it fast)
    model.eval()
    samples = model.generate(batch_size=2, steps=2)
    print(f"generated samples shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
