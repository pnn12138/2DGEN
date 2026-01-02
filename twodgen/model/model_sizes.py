from __future__ import annotations

from typing import Dict, Optional

MODEL_SIZE_PRESETS: Dict[str, Dict[str, float | int]] = {
    "tiny": {
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "time_embed_dim": 192,
        "z_embed_dim": 96,
        "f_embed_dim": 96,
        "rbf_dim": 24,
        "pair_mlp_hidden": 96,
    },
    "base": {
        "embed_dim": 256,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "time_embed_dim": 256,
        "z_embed_dim": 128,
        "f_embed_dim": 128,
        "rbf_dim": 32,
        "pair_mlp_hidden": 128,
    },
    "large": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "time_embed_dim": 384,
        "z_embed_dim": 192,
        "f_embed_dim": 192,
        "rbf_dim": 48,
        "pair_mlp_hidden": 192,
    },
    "xl": {
        "embed_dim": 512,
        "depth": 16,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "time_embed_dim": 512,
        "z_embed_dim": 256,
        "f_embed_dim": 256,
        "rbf_dim": 64,
        "pair_mlp_hidden": 256,
    },
}


def resolve_model_hparams(
    size: str, overrides: Optional[Dict[str, Optional[float | int]]] = None
) -> Dict[str, float | int]:
    if size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unknown model size preset: {size}")
    hparams = MODEL_SIZE_PRESETS[size].copy()
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                hparams[key] = value
    if int(hparams["embed_dim"]) % int(hparams["num_heads"]) != 0:
        raise ValueError(
            f"embed_dim ({hparams['embed_dim']}) must be divisible by num_heads ({hparams['num_heads']})."
        )
    return hparams


__all__ = ["MODEL_SIZE_PRESETS", "resolve_model_hparams"]
