# 2DGEN 子项目说明

## 内容概览
- 模型：`model/model.py` 提供小型 ViT (`C2DBVisionTransformer`) 与 JiT 风格扩散模型 (`C2DBJiT`)，`model/denoiser.py` 封装为 `C2DBDenoiser`，使用速度空间 (v-pred) 损失。
- 扩散与损失：`common/diffusion.py` 定义时间采样、标签 dropout、Clean/Velocity 预测损失。
- 数据：`data/c2db_dataset.py` 提供 CSV 解析版 `C2DBDataset` 与预处理后 npz 版 `C2DBGridNPZDataset`；`data/prepare_c2db_grid.py` 从 `data/C2DB/c2db_summary.csv` 生成 3×24×3 网格并保存 npz。
- 测试/训练：`test.py` 为随机张量烟雾测试；`train.py` 读取预处理好的 npz 进行简单训练并保存 checkpoint。

## 数据预处理
1. 准备原始 CSV：`data/C2DB/c2db_summary.csv`（已包含 CIF 文本）。
2. 生成网格 npz（默认 3×24×3，通道为原子序数/分数坐标/掩码）：
   ```bash
   uv run python 2DGEN/data/prepare_c2db_grid.py \
     --csv data/C2DB/c2db_summary.csv \
     --out data/C2DB/ache/c2db_grid.npz
   ```
   `--max-atoms` 控制最多保留的原子数；`--atomic-scale` 控制原子序数归一化。

## 训练
使用预处理好的 npz：
```bash
uv run python 2DGEN/train.py \
  --data data/C2DB/ache/c2db_grid.npz \
  --epochs 1 --batch-size 64 --lr 1e-4
```
- 自动选择 GPU/CPU；每个 epoch 保存 checkpoint 到 `outputs/checkpoints/`。
- 可选参数：`--num-workers`、`--log-interval`、`--max-steps`（限制总步数便于快速验证）。

## 快速自测
```bash
uv run python 2DGEN/test.py
```
执行随机输入的前向与反向传播，打印损失和张量形状。
