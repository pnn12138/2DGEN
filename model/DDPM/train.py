# train.py（替换训练循环核心部分）
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import CFG
from data import get_dataloader
from schedules import DiffusionSchedule
from unet import SimpleUNet

def train():
    device = CFG.device if torch.cuda.is_available() else "cpu"
    loader = get_dataloader(train=True)
    sched = DiffusionSchedule(CFG.timesteps, CFG.beta_schedule, device)
    net = SimpleUNet().to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=CFG.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    writer = SummaryWriter(log_dir="runs/cddpm")

    net.train(); gstep = 0
    for epoch in range(1, CFG.num_epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CFG.num_epochs}")
        for x0, y in pbar:
            x0, y = x0.to(device), y.to(device)              # y ∈ [0..9]
            bsz = x0.size(0)
            t = torch.randint(0, CFG.timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)

            # 前向扩散
            xt = sched.q_sample(x0, t, noise=noise)

            # classifier-free：随机把一部分标签置为 -1（走“无条件”）
            drop_mask = (torch.rand(bsz, device=device) < CFG.cond_drop_prob)
            y_train = y.clone()
            y_train[drop_mask] = -1

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                eps_pred = net(xt, t, y_train)              # 注意传入 y_train
                loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            gstep += 1
            pbar.set_postfix(loss=float(loss))
            if gstep % 50 == 0:
                writer.add_scalar("train/loss", float(loss), gstep)

        torch.save({"model": net.state_dict()}, "cddpm_mnist.pth")
        print("Saved checkpoint to cddpm_mnist.pth")

    writer.close()

if __name__ == "__main__":
    train()
