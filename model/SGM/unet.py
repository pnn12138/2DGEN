# 模块 C：UNet-Tiny（支持 ε-头 或 score-头），输入 [B, C, H, W]
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你的 TimeEmbedding（模块 B）
from model.SGM.time_emb import TimeEmbedding

class ResBlock(nn.Module):
    """
    残差块：GN → SiLU → Conv → (加时间侧支) → GN → SiLU → Conv
    时间侧支：h_t[ B,emb ] 通过线性映射成通道偏置，加到特征图上
    """
    def __init__(self, in_ch, out_ch, time_ch, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)  # 把时间特征映射到通道偏置
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, h_t):
        # x: [B,in_ch,H,W], h_t: [B,time_ch]
        B, _, H, W = x.shape
        out = self.conv1(F.silu(self.norm1(x)))
        # 时间侧支作为通道偏置加到特征图
        time_bias = self.time_proj(h_t).view(B, -1, 1, 1)
        out = out + time_bias
        out = self.conv2(F.silu(self.norm2(out)))
        return out + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):  # 下采样一半分辨率
        return self.op(x)

class Up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=1)
    def forward(self, x):  # 上采样×2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNetTiny(nn.Module):
    """
    一个很小的 U-Net：
      enc:  C -> ch -> 2ch -> 4ch
      dec:  4ch -> 2ch -> ch -> C
    每层含 1~2 个 ResBlock，均注入时间编码 h_t。
    head_type: "eps" 或 "score"
    """
    def __init__(self, in_ch=3, base_ch=64, time_dim=256, head_type="eps"):
        super().__init__()
        assert head_type in ["eps", "score"]
        self.head_type = head_type

        self.time_embed = TimeEmbedding(dim=128, out_dim=time_dim)

        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, base_ch, time_dim)
        self.down1 = Down(base_ch)               # -> base_ch, H/2
        self.rb2 = ResBlock(base_ch, base_ch*2, time_dim)
        self.down2 = Down(base_ch*2)            # -> 2base, H/4
        self.rb3 = ResBlock(base_ch*2, base_ch*4, time_dim)

        # Bottleneck
        self.rb_mid = ResBlock(base_ch*4, base_ch*4, time_dim)

        # Decoder（带 skip）
        self.up1 = Up(base_ch*4, base_ch*2)     # <- skip rb2
        self.rb4 = ResBlock(base_ch*4, base_ch*2, time_dim)
        self.up2 = Up(base_ch*2, base_ch)       # <- skip rb1
        self.rb5 = ResBlock(base_ch*2, base_ch, time_dim)

        # 头部：预测与输入同通道数
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B,C,H,W]
        t: [B] 或 [B,1]，范围(0,1]（与 SDE 的 t 保持一致）
        返回：与 x 同形状的预测（eps 或 score）
        """
        h_t = self.time_embed(t)  # [B,time_dim]

        # enc
        x0 = self.in_conv(x)                # [B,base,H,W]
        e1 = self.rb1(x0, h_t)              # skip-1
        d1 = self.down1(e1)                 # [B,base,H/2,W/2]
        e2 = self.rb2(d1, h_t)              # skip-2  channels=2base
        d2 = self.down2(e2)                 # [B,2base,H/4,W/4]
        e3 = self.rb3(d2, h_t)              # [B,4base,H/4,W/4]

        # mid
        m = self.rb_mid(e3, h_t)            # [B,4base,H/4,W/4]

        # dec
        u1 = self.up1(m)                    # [B,2base,H/2,W/2]
        u1 = torch.cat([u1, e2], dim=1)     # skip 接回
        u1 = self.rb4(u1, h_t)              # [B,2base,H/2,W/2]

        u2 = self.up2(u1)                   # [B,base,H,W]
        u2 = torch.cat([u2, e1], dim=1)     # skip 接回
        u2 = self.rb5(u2, h_t)              # [B,base,H,W]

        out = self.out_conv(F.silu(self.out_norm(u2)))  # [B,C,H,W]
        return out  # 语义由 head_type 决定：eps 或 score
