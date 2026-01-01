# data.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import CFG

def get_dataloader(train: bool = True):
    """
    返回 MNIST 的 DataLoader：
      - 图像尺寸：CFG.image_size（默认 28）
      - 归一化到 [-1, 1]，与 DDPM 公式/采样反归一化一致
    """
    tfm = transforms.Compose([
        transforms.Resize(CFG.image_size),
        transforms.ToTensor(),                          # [0,1]
        transforms.Lambda(lambda x: x * 2. - 1.)        # → [-1,1]
    ])
    ds = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=tfm
    )
    loader = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True
    )
    return loader
