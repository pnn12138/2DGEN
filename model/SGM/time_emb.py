# 模块 B：时间编码（通用于 VP/VE）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """
    把标量时间 t 映射到 dim 维的 sin/cos 向量。
    t: [B] 或 [B, 1]，范围 (0,1]（建议）
    返回: [B, dim]
    """
    # 生成频率带
    half = dim // 2
    # 指数频率：freqs ~ exp(-log(max_period) * i/half)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=t.dtype) / half
    )
    # [B, 1] × [half] -> [B, half]
    args = t.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(2 * math.pi * args), torch.cos(2 * math.pi * args)], dim=-1)  # [B, 2*half]
    if dim % 2 == 1:  # 若维度为奇数，补齐
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # [B, dim]

class TimeEmbedding(nn.Module):
    """
    Sinusoidal 编码 + 两层 MLP 投影到需要的通道数。
    用法：
      t_embed = TimeEmbedding(dim=128, out_dim=256)
      h_t = t_embed(t)   # [B, 256]
    """
    def __init__(self, dim: int = 128, out_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor):
        # t: [B] or [B,1] in (0,1]
        emb = timestep_embedding(t.view(-1), self.dim)       # [B, dim]
        h = F.silu(self.fc1(emb))                            # [B, out_dim]
        h = self.dropout(F.silu(self.fc2(h)))                # [B, out_dim]
        return h
