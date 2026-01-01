# train_vp_mnist.py
import os
from pathlib import Path
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# === 你的模块 ===
from model.SGM.SDE import SDE, SDEConfig
from model.SGM.unet import UNetTiny
from model.SGM.VP_loss import VPEpsilonLoss, VPTrainConfig
# ===============

def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v.copy_(self.shadow[k])

def build_loader(data_root: str, batch_size: int, num_workers: int):
    import torchvision
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize((0.5,), (0.5,)),     # -> [-1,1]
    ])
    # 你截图里已下载成 raw/ 结构，下面 download=False 就行
    ds = torchvision.datasets.MNIST(root=data_root, train=True, transform=tfm, download=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=True)
    return loader

def save_ckpt(path: Path, step: int, model: nn.Module, opt, ema: Optional[EMA], scaler: Optional[torch.cuda.amp.GradScaler], sde_cfg: SDEConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "ema": (ema.shadow if ema is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "sde_cfg": sde_cfg.__dict__,
    }
    torch.save(ckpt, path)
    print(f"[ckpt] saved -> {path}")

def load_ckpt(path: str, model: nn.Module, opt=None, ema: Optional[EMA]=None, scaler: Optional[torch.cuda.amp.GradScaler]=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.shadow = ckpt["ema"]
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    print(f"[ckpt] loaded from {path}, step={step}")
    return step

def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # data
    loader = build_loader(args.data_root, args.batch_size, args.num_workers)

    # SDE (VP)
    sde_cfg = SDEConfig(sde_type="VP", beta_min=args.beta_min, beta_max=args.beta_max, t_epsilon=1e-5)
    sde = SDE(sde_cfg)

    # model
    net = UNetTiny(in_ch=1, base_ch=args.base_ch, time_dim=args.time_dim, head_type="eps").to(device)

    # loss
    loss_cfg = VPTrainConfig(loss_weighting=args.loss_weighting)
    criterion = VPEpsilonLoss(sde, loss_cfg)

    # opt / ema / amp
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    ema = EMA(net, decay=args.ema) if args.ema > 0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # resume
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        global_step = load_ckpt(args.resume, net, opt, ema, scaler)

    # log
    writer = SummaryWriter(log_dir=args.log_dir)
    save_dir = Path(args.save_dir)

    # loop
    net.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for (x, _) in pbar:
            x = x.to(device, non_blocking=True)  # [B,1,28,28], 已归一化到[-1,1]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                loss, logs = criterion(net, x)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()

            if ema is not None:
                ema.update(net)

            global_step += 1
            pbar.set_postfix(loss=float(loss))

            # log
            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", float(loss), global_step)
                writer.add_scalar("train/mse_unweighted", float(logs["mse_unweighted"]), global_step)
                writer.add_scalar("train/t_mean", float(logs["t_mean"]), global_step)
                writer.add_scalar("train/sigma_mean", float(logs["sigma_mean"]), global_step)

            # ckpt
            if global_step % args.ckpt_every == 0:
                save_ckpt(save_dir / f"vp_step_{global_step}.pt", global_step, net, opt, ema, scaler, sde_cfg)
                if ema is not None:
                    # 另存一份 EMA 权重
                    raw = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    ema.apply_to(net)
                    save_ckpt(save_dir / "ema" / f"vp_ema_step_{global_step}.pt", global_step, net, opt, ema, scaler, sde_cfg)
                    net.load_state_dict(raw, strict=True)

    writer.close()
    print("Training finished.")

def build_args():
    p = argparse.ArgumentParser("MNIST | VP-SGM (eps) trainer")
    # data
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    # model
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    # sde (vp)
    p.add_argument("--beta_min", type=float, default=0.1)
    p.add_argument("--beta_max", type=float, default=20.0)
    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--loss_weighting", type=str, default="none", choices=["none", "sigma", "snr"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # io
    p.add_argument("--save_dir", type=str, default="./checkpoints_vp_mnist")
    p.add_argument("--log_dir", type=str, default="./runs/vp_mnist")
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", type=str, default="")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = build_args()
    train(args)
