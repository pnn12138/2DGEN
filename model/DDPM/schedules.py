# schedules.py
import math
import torch
from config import CFG

# -------- β 调度（决定每一步往前扩散时加多少噪声） --------
def linear_beta_schedule(timesteps: int):
    """
    线性 β 调度：β_t 从很小线性增大到较大。
    优点：实现简单，经典基线；缺点：在高步数末段可能过噪或训练不稳。
    """
    beta_start, beta_end = 1e-4, 2e-2
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    余弦 ᾱ(t) 调度（改进DDPM：Nichol & Dhariwal, 2021）
    思想：先在连续时间上设计平滑的 ᾱ(t)=cos^2(...) 曲线，再离散为 β_t。
    通常带来更稳定训练、更好的样本质量。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]               # 归一化，使 ᾱ_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])            # β_t = 1 - ᾱ_t / ᾱ_{t-1}
    betas = torch.clip(betas, 1e-8, 0.999)                             # 数值稳定
    return betas.float()

def make_beta_schedule(timesteps: int, kind: str):
    if kind == "linear":
        return linear_beta_schedule(timesteps)
    elif kind == "cosine":
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown beta_schedule: {kind}")

# -------- 封装：所有跟 q(x_t|x_0) 和采样后验系数有关的量 --------
class DiffusionSchedule:
    """
    职责：
    1) 由 β_t 计算 α_t=1-β_t、ᾱ_t=∏_{i=1}^t α_i 等核心表；
    2) 提供前向扩散采样 q_sample(x_t|x_0)；
    3) 预先计算反向采样用的“后验方差”posterior_variance 等系数（供采样模块用）。
    """
    def __init__(self, timesteps=CFG.timesteps, kind=CFG.beta_schedule, device="cpu"):
        betas = make_beta_schedule(timesteps, kind).to(device)         # (T,)
        alphas = 1.0 - betas                                           # α_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)                  # ᾱ_t
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )

        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # 常用开方项（实现 q(x_t|x_0) 和 μ_θ 时会频繁用到）
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                     # √ᾱ_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)       # √(1-ᾱ_t)
        self.one_over_sqrt_alpha = torch.rsqrt(alphas)                             # 1/√α_t

        # 反向采样需要的 q(x_{t-1}|x_t,x_0) 的方差项（论文附录闭式）
        # Var = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        前向扩散采样：q(x_t | x_0) = √ᾱ_t * x0 + √(1-ᾱ_t) * ε，其中 ε~N(0,I)
        作用：给定原图 x0 和时间步 t，把它“加噪”到 x_t，用于训练时的监督。
        形状约定：
          - x0: (B, C, H, W)
          - t : (B,) 每个样本自己的时间步
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # 按 batch 取对应 t 的系数，并广播到 (B,C,H,W)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    betas = linear_beta_schedule(100).detach().cpu().numpy()
    plt.figure()
    plt.plot(betas, marker='o')
    plt.title("linear beta schedule (10)")
    plt.xlabel("t")
    plt.ylabel("beta_t")
    plt.grid(True)
    plt.savefig("beta_schedule.png", dpi=150, bbox_inches="tight")
    print("Saved plot to beta_schedule.png")
