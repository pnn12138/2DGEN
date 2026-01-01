# config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 数据与模型
    image_size: int = 28          # 训练图像的边长（MNIST=28）
    channels: int = 1             # 通道数（灰度=1，彩色=3）
    timesteps: int = 1000         # DDPM时间步数 T
    beta_schedule: str = "cosine" # β调度："linear" 或 "cosine"

    # 训练
    batch_size: int = 2048
    lr: float = 2e-4
    num_epochs: int = 100
    device: str = "cuda"          # "cuda" 或 "cpu"

    # 采样
    sample_steps: int = 1000       # 采样使用的步数（可小于 timesteps 做跳步采样）

    # 在 TrainConfig 中新增
    num_classes: int = 10          # MNIST=10
    cond_drop_prob: float = 0.1    # 训练时丢标签做无条件分支的概率
    guidance_scale: float = 2.0    # 采样时的 classifier-free guidance 强度

CFG = TrainConfig()

def exists(x):
    return x is not None
