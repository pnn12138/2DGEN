# unet.py
import torch
import torch.nn as nn
from layers import ResBlock, TimeMLP, sinusoidal_time_embedding, LabelEmbedding
from config import CFG

class SimpleUNet(nn.Module):
    """
    升级：支持类条件 y。无条件分支通过 LabelEmbedding 中的 null_id 实现。
    """
    def __init__(self, in_ch=CFG.channels, base_ch=64, t_dim=128, y_dim=128, num_classes=CFG.num_classes):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = TimeMLP(t_dim, t_dim)
        self.label_emb = LabelEmbedding(num_classes, y_dim)
        # 把 t 与 y 的嵌入融合到同一维度（加和前先线性到同一维度）
        self.to_t = nn.Linear(t_dim, t_dim)
        self.to_y = nn.Linear(y_dim, t_dim)

        # encoder
        self.enc1 = ResBlock(in_ch, base_ch, t_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)

        self.enc2 = ResBlock(base_ch, base_ch * 2, t_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)

        # bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, t_dim)

        # decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.dec1 = ResBlock(base_ch * 4, base_ch, t_dim)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)
        self.dec2 = ResBlock(base_ch * 2, base_ch, t_dim)

        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def fuse_cond(self, t_emb, y_emb):
        # 把 t/y 投影到同一隐空间后相加（也可 concat 后再线性）
        return self.to_t(t_emb) + self.to_y(y_emb)

    def forward(self, x, t, y=None):
        # 时间嵌入
        t_emb = sinusoidal_time_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)

        # 标签嵌入（y 可能为 None 或含 -1 表示无条件）
        if y is None:
            # 若未提供，全部走无条件
            y = torch.full_like(t, -1)
        y_emb = self.label_emb(y)

        # 融合条件
        cond = self.fuse_cond(t_emb, y_emb)  # (B, t_dim)

        # 编码-瓶颈-解码（与无条件相同，但把 cond 传入每个 ResBlock）
        e1 = self.enc1(x, cond)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, cond)
        d2 = self.down2(e2)

        m = self.mid(d2, cond)

        u1 = self.up1(m)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.dec1(u1, cond)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.dec2(u2, cond)

        return self.out(u2)
