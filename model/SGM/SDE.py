# 模块 A：统一的 SDE 封装（VP / VE）
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SDEConfig:
    sde_type: str = "VP"   # "VP" 或 "VE"
    beta_min: float = 0.1  # VP: β(t) = β_min + t(β_max-β_min)
    beta_max: float = 20.0
    sigma_min: float = 0.01  # VE: σ(t) = σ_min * (σ_max/σ_min)^t
    sigma_max: float = 50.0
    t_epsilon: float = 1e-5 # 避免 t=0 数值问题

class SDE:
    def __init__(self, cfg: SDEConfig):
        self.cfg = cfg
        if cfg.sde_type not in ["VP", "VE"]:
            raise ValueError("sde_type must be 'VP' or 'VE'")

    # ===== VP-SDE 定义 =====
    # dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
    # 边际：x(t) = x0 * exp(-0.5 ∫β) + N(0, 1 - exp(-∫β))
    def _vp_beta(self, t):
        return self.cfg.beta_min + t * (self.cfg.beta_max - self.cfg.beta_min)

    def _vp_mean_std(self, x0, t):
        # 解析积分 ∫0^t β(s) ds = β_min * t + 0.5*(β_max-β_min)*t^2
        beta_min, beta_max = self.cfg.beta_min, self.cfg.beta_max
        integral = beta_min * t + 0.5 * (beta_max - beta_min) * (t ** 2)
        mean = torch.exp(-0.5 * integral).view(-1, 1, 1, 1) * x0
        var = 1.0 - torch.exp(-integral).view(-1, 1, 1, 1)
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        return mean, std

    # ===== VE-SDE 定义 =====
    # dx = 0 * dt + sigma(t) * sqrt{2 log(σ_max/σ_min)} dW（常见参数化之一）
    # 边际：x(t) = x0 + N(0, σ(t)^2 - σ_min^2)（或直接用 σ(t) 视作扰动尺度）
    def _ve_sig·ma(self, t):
        # 几何插值：σ(t) = σ_min * (σ_max/σ_min)^t
        return self.cfg.sigma_min * (self.cfg.sigma_max / self.cfg.sigma_min) ** t

    def _ve_mean_std(self, x0, t):
        # 常用训练里，我们不衰减均值，直接把 σ(t) 当作噪声尺度采样 z~N(0, I)
        # 一种简化的“边际 std”：std = σ(t)
        # 若你更偏好 NCSN++ 里的严格推导，也可以改用 std = sqrt(σ(t)^2 - σ(0)^2)
        std = self._ve_sigma(t).view(-1, 1, 1, 1)
        mean = x0  # 漂移为 0
        return mean, std

    # 统一接口：边际分布（训练时会直接采样 x_t = mean + std * z）
    def marginal_prob(self, x0, t):
        if self.cfg.sde_type == "VP":
            return self._vp_mean_std(x0, t)
        else:
            return self._ve_mean_std(x0, t)

    # 统一接口：前向 SDE 的 f, g
    def sde(self, x, t):
        t = t.clamp(min=self.cfg.t_epsilon, max=1.0 - 1e-7)
        if self.cfg.sde_type == "VP":
            beta_t = self._vp_beta(t).view(-1, 1, 1, 1)
            drift = -0.5 * beta_t * x
            diffusion = torch.sqrt(torch.clamp(beta_t, min=1e-12))
            return drift, diffusion
        else:  # VE
            # 常用一个常数扩散强度（与 log(σ_max/σ_min) 有关），这里给出简化版本
            # 你也可以让 g(t)=sigma(t)*sqrt(2*log(...))；学习上差别不大
            sigma_t = self._ve_sigma(t).view(-1, 1, 1, 1)
            drift = torch.zeros_like(x)
            diffusion = sigma_t * 0 + 1.0  # 简化为常数，便于数值稳定起步
            return drift, diffusion

    # 终端先验采样（t=1）
    def prior_sampling(self, shape, device=None):
        return torch.randn(*shape, device=device)
