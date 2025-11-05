# 模块 D：VP + ε-头 的加噪与损失
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from model.SGM.SDE import SDE
# 假设你已引入：
# from your_sde import SDE, SDEConfig
# from your_unet import UNetTiny

@dataclass
class VPTrainConfig:
    t_min: float = 1e-5      # 避开 t=0
    t_max: float = 1.0
    loss_weighting: str = "none"  # 可选: "none" | "snr" | "sigma"
    # "none": 纯 MSE
    # "snr": 以 SNR(t) = (mu/std)^2 为权重的变体（可改善高 t 稳定性）
    # "sigma": 以 1/sigma(t)^2 加权（等价于直接匹配 score）

class VPEpsilonLoss(nn.Module):
    """
    给定 (x0, t)，构造 xt = mu + sigma*eps，并回归 eps。
    可选权重: none/snr/sigma。
    """
    def __init__(self, sde: SDE, cfg: VPTrainConfig = VPTrainConfig()):
        super().__init__()
        self.sde = sde
        self.cfg = cfg

    @torch.no_grad()
    def sample_t(self, B, device):
        # 常用 Uniform(0,1]；你也可以用 cos / 余弦密度重采样等
        t = torch.rand(B, device=device) * (self.cfg.t_max - self.cfg.t_min) + self.cfg.t_min
        return t

    def forward(self, net, x0):
        """
        x0: [B,C,H,W] 预处理到 [-1,1] 或零均值单位方差
        返回: loss, 以及监控用的 dict
        """
        B = x0.size(0)
        device = x0.device

        # 1) 采样时间
        t = self.sample_t(B, device)  # [B]

        # 2) 计算边际均值/标准差（来自 VP SDE）
        mu, sigma = self.sde.marginal_prob(x0, t)  # mu: [B,...], sigma: [B,1,1,1]

        # 3) 采样噪声，构造 xt
        eps = torch.randn_like(x0)
        xt = mu + sigma * eps

        # 4) 前向，预测 eps
        eps_hat = net(xt, t)  # [B,C,H,W]

        # 5) 可选加权
        if self.cfg.loss_weighting == "none":
            w = 1.0
        elif self.cfg.loss_weighting == "sigma":
            # 等价回归 score：|| eps_hat - eps ||^2 / sigma^2
            w = 1.0 / (sigma**2 + 1e-12)  # broadcast 到 [B,1,1,1]
        elif self.cfg.loss_weighting == "snr":
            # SNR = (mu/std)^2 = (e^{-0.5Λ} / sqrt(1-e^{-Λ}))^2 = e^{-Λ}/(1-e^{-Λ})
            # 用 SNR + 1 或其变体。这里给个简单版：1 + SNR
            alpha_bar = (mu / (x0 + 1e-12)).abs()  # 近似 e^{-0.5Λ(t)} 的幅度（逐元素），更稳可重算 Λ
            # 更稳健写法（推荐）：直接重算 Λ(t)
            beta_min, beta_max = self.sde.cfg.beta_min, self.sde.cfg.beta_max
            Lambda = beta_min * t + 0.5 * (beta_max - beta_min) * (t**2)  # [B]
            alpha_bar_true = torch.exp(-Lambda).view(B, 1, 1, 1)
            snr = alpha_bar_true / (1.0 - alpha_bar_true + 1e-12)
            w = 1.0 + snr
        else:
            raise ValueError("unknown loss_weighting")

        # 6) MSE
        loss = (w * (eps_hat - eps) ** 2).mean()

        # 一些监控量
        with torch.no_grad():
            mse_unweighted = F.mse_loss(eps_hat, eps)
            sigma_mean = sigma.mean()
        logs = {
            "mse_unweighted": mse_unweighted.detach(),
            "sigma_mean": sigma_mean.detach(),
            "t_mean": t.mean().detach()
        }
        return loss, logs
