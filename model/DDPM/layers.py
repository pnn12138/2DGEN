# layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) 正弦时间嵌入 (Sinusoidal Time Embedding)
# -----------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    把离散时间步 t (B,) 映射到连续向量 (B, dim)
    形式与 Transformer 的位置编码一致：sin/cos 多频率基
    """
    device = t.device
    half = dim // 2
    # 频率按指数间隔分布，覆盖多尺度
    freq = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / max(half - 1, 1)))
    angles = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:  # 若 dim 为奇数，补齐 1 维
        emb = F.pad(emb, (0, 1))
    return emb
 
# -----------------------------
# 2) 时间条件 MLP (把时间嵌入调制进通道维度)
# -----------------------------
class TimeMLP(nn.Module):
    """
    作用：把时间嵌入 emb -> 一个与通道数对齐的向量，用于调制卷积特征
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),               # 更平滑的非线性，扩散模型常用
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)  # (B, out_dim)

# -----------------------------
# 3) 残差块（时间调制版）
# -----------------------------
class ResBlock(nn.Module):
    """
    时间条件残差块：
      x -----+-----------------------> (+) ----> SiLU
             |                        ^
             |                        |
             Conv3x3 -> SiLU -> +t   Conv3x3
                                  |
                                Linear(t_emb)->(B,out_ch)->(B,out_ch,1,1)

    - 输入:  (B, C_in, H, W), t_emb: (B, t_dim)
    - 输出:  (B, C_out, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # 用一层线性把时间嵌入转成 “通道偏置”，加到 conv1 后的特征图上
        self.time_to_channel = nn.Linear(t_dim, out_ch)

        # 对齐残差分支通道数
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # 可选：归一化（小模型上可省略/也可换成 GroupNorm）
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # time conditioning: 把 (B,out_ch) 变成 (B,out_ch,1,1) 后加到特征图
        t_feat = self.time_to_channel(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_feat

        h = self.conv2(h)
        h = self.norm2(h)
        # 残差连接
        return self.act(h + self.skip(x))

# 在文件顶部保留原有 import

#条件嵌入
class LabelEmbedding(nn.Module):
    """
    把离散类别 y -> 向量 (dim)。index=-1 代表“无条件”分支（专用可学习向量）。
    """
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, dim)  # 额外 +1 作为 "null" 类别
        self.null_id = num_classes  # 作为无条件 id

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,) 取值 0..num_classes-1 或 -1 表示无条件
        """
        y = y.clone()
        y[y < 0] = self.null_id
        return self.emb(y)  # (B, dim)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    T = 200                    # 试着看 0~199 这些时间步
    dim = 128                  # 嵌入维度
    t = torch.arange(T)
    emb = sinusoidal_time_embedding(t, dim).numpy()

    plt.figure(figsize=(10, 4))
    for i in range(8):         # 画前 8 个通道
        plt.plot(t, emb[:, i], label=f"dim {i}")
    plt.legend(ncol=4, fontsize="small")
    plt.title("Sinusoidal Time Embedding (first 8 dims)")
    plt.xlabel("timestep")
    plt.ylabel("embedding value")
    plt.tight_layout()
    plt.show()