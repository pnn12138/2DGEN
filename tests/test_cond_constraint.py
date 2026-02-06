import torch
import types

from twodgen.common.atom_diffusion import AtomDiffusionConfig, AtomVelocityLoss


class _DummyModel(torch.nn.Module):
    def __init__(self, g_scale: float = 1.0, num_elements: int = 118) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(
            g_scale=g_scale,
            diffusion=types.SimpleNamespace(cond_max=None),
            model=types.SimpleNamespace(pbc_mask=(1, 1, 1), g_scale=g_scale),
        )
        self.num_elements = num_elements

    def forward(
        self,
        z_masked,
        frac_t,
        cell_t,
        atom_mask,
        t,
        cond_in,
        counts_in,
        uv_angle=None,
        z_norm=None,
        lattice_param=None,
        slab_t=None,
        return_geom: bool = False,
        **kwargs,
    ):
        logits_z = torch.zeros(
            (z_masked.shape[0], z_masked.shape[1], self.num_elements + 1),
            device=z_masked.device,
            dtype=frac_t.dtype,
        )
        if return_geom:
            geom = types.SimpleNamespace(
                uv_angle=None,
                z_norm=None,
                lattice_param=None,
                t=None,
            )
            return frac_t, cell_t, logits_z, geom
        return frac_t, cell_t, logits_z


def _run_loss(gram6, cond_max):
    cfg = AtomDiffusionConfig(
        lambda_cond=0.1,
        cond_max=cond_max,
        lambda_angle=0.0,
        lambda_chol_bound=0.0,
        lambda_expand_collision=0.0,
        lambda_vacuum=0.0,
        lambda_volume=0.0,
        lambda_c_len=0.0,
        lambda_anisotropy=0.0,
        lambda_cross_vacuum=0.0,
        cell_rep="gram6",
    )
    loss_fn = AtomVelocityLoss(cfg, mask_token_id=119)
    model = _DummyModel()
    z = torch.tensor([[1, 2]], dtype=torch.long)
    frac = torch.zeros((1, 2, 3), dtype=torch.float32)
    atom_mask = torch.ones((1, 2), dtype=torch.float32)
    cond = torch.zeros((1, 1), dtype=torch.float32)
    counts_vector = torch.zeros((1, 118), dtype=torch.float32)
    loss, _, _, _, metrics = loss_fn(
        model,
        z,
        frac,
        atom_mask,
        gram6,
        cond,
        counts_vector,
        uv_angle=None,
        z_norm=None,
        lattice_param=None,
        slab_t=None,
        min_dist_train_weight=0.0,
        min_dist_train_cut=1.5,
    )
    return loss, metrics


def test_cond_penalty_triggers_on_bad_lattice():
    # Extremely ill-conditioned lattice: diag(1000, 1e-3, 1)
    gram6_bad = torch.tensor([[1e6, 1e-6, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    _, metrics = _run_loss(gram6_bad, cond_max=10.0)
    assert metrics["loss_cond_number"] > 0
    assert torch.isfinite(metrics["cond_gram_mean"])
    assert torch.isfinite(metrics["cond_lattice_mean"])
    assert torch.isfinite(metrics["cond_diff_abs_mean"])
    assert torch.isfinite(metrics["cond_valid_rate"])


def test_cond_penalty_zero_on_good_lattice():
    gram6_good = torch.tensor([[4.0, 4.0, 4.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    _, metrics = _run_loss(gram6_good, cond_max=1e6)
    assert metrics["loss_cond_number"] < 1e-6
    assert torch.isfinite(metrics["cond_gram_mean"])
    assert torch.isfinite(metrics["cond_lattice_mean"])
    assert torch.isfinite(metrics["cond_diff_abs_mean"])
    assert torch.isfinite(metrics["cond_valid_rate"])
