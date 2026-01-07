"""
Smoke test for token-based crystal diffusion.
"""

import torch
from twodgen.model.atom_denoiser import AtomDenoiser


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AtomDenoiser().to(device)

    bsz = 2
    n = 8
    max_atoms = 12
    z = torch.zeros(bsz, max_atoms, dtype=torch.long, device=device)
    z[:, :n] = torch.randint(1, 10, (bsz, n), device=device)
    frac = torch.rand(bsz, max_atoms, 3, device=device)
    atom_mask = torch.zeros(bsz, max_atoms, device=device)
    atom_mask[:, :n] = 1.0
    gram6 = torch.randn(bsz, 6, device=device)

    loss, pred_v_f, pred_v_g, logits_z, _ = model(z, frac, atom_mask, gram6)
    loss.backward()
    print(f"loss: {loss.item():.4f}")
    print(f"pred_v_f: {tuple(pred_v_f.shape)} pred_v_g: {tuple(pred_v_g.shape)} logits_z: {tuple(logits_z.shape)}")

    model.eval()
    z_s, frac_s, gram_s, mask_s, lat_s, t_s, _, _ = model.generate(
        num_atoms=4, max_atoms=8, batch_size=2, steps=2
    )
    print(
        f"samples z: {tuple(z_s.shape)} frac: {tuple(frac_s.shape)} "
        f"gram: {tuple(gram_s.shape)} mask: {tuple(mask_s.shape)} "
        f"lat: {None if lat_s is None else tuple(lat_s.shape)} "
        f"t: {None if t_s is None else tuple(t_s.shape)}"
    )


if __name__ == "__main__":
    main()
