"""
Smoke tests for geometry heads, dual-graph, wrap embedding, and thickness head.
"""

from __future__ import annotations

import torch

from twodgen.common.crystal import lattice_to_gram6
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.model.atom_transformer import AtomTransformerConfig


def _build_lattice(batch_size: int, device: torch.device) -> torch.Tensor:
    base = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    noise = torch.randn(batch_size, 3, 3, device=device) * 0.05
    lattice = base + noise
    lattice[:, 0, 0] = lattice[:, 0, 0].abs() + 1.0
    lattice[:, 1, 1] = lattice[:, 1, 1].abs() + 1.0
    lattice[:, 2, 2] = lattice[:, 2, 2].abs() + 1.0
    return lattice


def _build_uv_angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.cos(2.0 * torch.pi * u),
            torch.sin(2.0 * torch.pi * u),
            torch.cos(2.0 * torch.pi * v),
            torch.sin(2.0 * torch.pi * v),
        ],
        dim=-1,
    )


def test_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, n, max_atoms = 2, 6, 8
    z = torch.zeros(bsz, max_atoms, dtype=torch.long, device=device)
    z[:, :n] = torch.randint(1, 10, (bsz, n), device=device)
    frac = torch.rand(bsz, max_atoms, 3, device=device)
    atom_mask = torch.zeros(bsz, max_atoms, device=device)
    atom_mask[:, :n] = 1.0

    lattice = _build_lattice(bsz, device)
    gram6 = lattice_to_gram6(lattice)

    u = torch.rand(bsz, max_atoms, device=device)
    v = torch.rand(bsz, max_atoms, device=device)
    uv_angle = _build_uv_angle(u, v)
    z_norm = torch.randn(bsz, max_atoms, device=device)
    lattice_param = torch.randn(bsz, 3, device=device)
    slab_t = torch.rand(bsz, device=device) + 0.5

    model_cfg = AtomTransformerConfig(
        num_elements=118,
        k_neighbors=8,
        g_scale=1.0,
        dual_graph=True,
        edge_type_dim=4,
        edge_type_gating=True,
        wrap_embed_dim=4,
        pbc_mask=(1, 1, 0),
    )
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    model = AtomDenoiser(denoiser_cfg).to(device)

    loss, _, _, _, metrics = model(
        z,
        frac,
        atom_mask,
        gram6,
        uv_angle=uv_angle,
        z_norm=z_norm,
        lattice_param=lattice_param,
        slab_t=slab_t,
    )
    loss.backward()
    for key in ("loss_uv", "loss_zn", "loss_lat", "loss_t"):
        if key not in metrics:
            raise AssertionError(f"Missing metric {key}")
    print("forward: ok")


def test_generate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = AtomTransformerConfig(
        num_elements=118,
        k_neighbors=8,
        g_scale=1.0,
        dual_graph=True,
        edge_type_dim=4,
        edge_type_gating=True,
        wrap_embed_dim=4,
        pbc_mask=(1, 1, 0),
    )
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    denoiser_cfg.project_geometry = True
    model = AtomDenoiser(denoiser_cfg).to(device)
    z_s, frac_s, gram_s, mask_s, lat_s, t_s = model.generate(num_atoms=4, max_atoms=6, batch_size=2, steps=2)
    print(
        f"generate: z={tuple(z_s.shape)} frac={tuple(frac_s.shape)} "
        f"gram={tuple(gram_s.shape)} mask={tuple(mask_s.shape)} "
        f"lat={None if lat_s is None else tuple(lat_s.shape)} "
        f"t={None if t_s is None else tuple(t_s.shape)}"
    )


def main() -> None:
    test_forward()
    test_generate()


if __name__ == "__main__":
    main()
