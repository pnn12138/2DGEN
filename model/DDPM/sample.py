# sample.py
import torch
from torchvision.utils import save_image, make_grid
from config import CFG
from schedules import DiffusionSchedule
from unet import SimpleUNet

@torch.no_grad()
def p_mean_variance_cfg(net, sched, x_t, t, y, guidance_scale: float):
    """
    CFG 版本的均值与方差：
      eps_uncond = εθ(x_t, t, y=-1)
      eps_cond   = εθ(x_t, t, y=y)
      eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    再用 eps_guided 构造 μθ 与 Var_post。
    """
    b = x_t.size(0)
    # 无条件与有条件各跑一次
    eps_uncond = net(x_t, t, torch.full_like(t, -1))
    eps_cond   = net(x_t, t, y)
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    beta_t  = sched.betas[t].view(b, 1, 1, 1)
    inv_sqrt_alpha_t = sched.one_over_sqrt_alpha[t].view(b, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sched.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)

    mean = inv_sqrt_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)
    var  = sched.posterior_variance[t].view(b, 1, 1, 1)
    return mean, var

@torch.no_grad()
def p_sample_step_cfg(net, sched, x_t, t, y, guidance_scale: float):
    mean, var = p_mean_variance_cfg(net, sched, x_t, t, y, guidance_scale)
    nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
    noise = torch.randn_like(x_t)
    return mean + nonzero_mask * torch.sqrt(var) * noise

@torch.no_grad()
def sample_conditional(n=64, cls=0, ckpt_path="cddpm_mnist.pth", guidance_scale=CFG.guidance_scale):
    """
    生成指定类别的样本（cls ∈ [0..9]）
    """
    device = CFG.device if torch.cuda.is_available() else "cpu"
    sched = DiffusionSchedule(CFG.timesteps, CFG.beta_schedule, device)
    net = SimpleUNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["model"])
    net.eval()

    x_t = torch.randn(n, CFG.channels, CFG.image_size, CFG.image_size, device=device)

    # 目标类别张量
    y = torch.full((n,), int(cls), device=device, dtype=torch.long)

    steps = CFG.sample_steps
    indices = torch.linspace(0, CFG.timesteps - 1, steps, dtype=torch.long, device=device)

    for i in reversed(indices):
        t = torch.full((n,), i.item(), device=device, dtype=torch.long)
        x_t = p_sample_step_cfg(net, sched, x_t, t, y, guidance_scale)

    x0 = (x_t.clamp(-1, 1) + 1) / 2.0
    grid = make_grid(x0, nrow=int(n ** 0.5))
    save_image(grid, f"samples_cls{cls}_gs{guidance_scale:.1f}.png")
    print(f"Saved samples_cls{cls}_gs{guidance_scale:.1f}.png")

if __name__ == "__main__":
    # 示例：采样“数字 3”，guidance_scale=2.0
    sample_conditional(n=64, cls=3, guidance_scale=2.0)
