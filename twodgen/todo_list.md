# twodgen 优化器与学习率策略（规划）

## 目标
在 token 扩散训练中提升稳定性与可复现性，统一优化器与学习率策略。

## 首选方案（稳健通用）
### 1) Optimizer
- 使用 `AdamW`
- 初始学习率 `1e-4`（batch_size=256 时的参考值）
- betas：`(0.9, 0.95)`（与 Transformer 训练习惯一致）
- 权重衰减 `1e-2`（小数据或不稳定时降到 `1e-3`）
- 对以下参数不做 weight decay（参数分组）：
  - `bias`
  - `LayerNorm`/`RMSNorm` 权重
  - 任何显式标量噪声权重（如 `s_f/s_g/s_z`）
  - 可选：embedding 表
- 可选：梯度裁剪 `clip_norm=1.0`（防止早期发散）

### 2) Scheduler
- warmup + cosine decay
  - warmup：前 `5%` 总 steps 线性升温到目标学习率
  - cosine：从目标学习率衰减到 `min_lr=1e-6`（或 `min_lr=lr*1e-2`）
- 以 step 为粒度更新（适配扩散训练）；若只按 epoch 更新，需保证每个 epoch 步数稳定
- 当 `--max-steps` 启用时，调度器以 `max_steps` 为总步数

### 3) EMA（可选）
- 使用参数 EMA（exponential moving average）
- 建议 decay `0.999`（短训练）或 `0.9999`（长训练）
- 采样阶段优先使用 EMA 权重
- 保存 EMA 权重到 checkpoint，采样脚本支持自动加载

## 实施要点（后续落地时）
- 在 `train_tokens.py` 增加优化器/调度器相关 CLI：
  - `--optimizer adamw`
  - `--lr 1e-4`
  - `--weight-decay 1e-2`
  - `--warmup-steps N`
  - `--min-lr 1e-6`
  - `--lr-schedule cosine`
  - `--betas 0.9,0.95`
  - `--clip-grad 1.0`
  - `--ema 0|1 --ema-decay 0.9999`
- 抽出 param group：`bias/LayerNorm` 不做 decay
- lr 线性缩放建议：`lr = 1e-4 * (global_batch/256)`
- 在 checkpoint 里记录 optimizer/scheduler/EMA 状态
- README 更新训练建议与默认配置

## 为什么选该方案
- AdamW + cosine 是 Transformer/扩散模型的稳态组合
- warmup 缓解早期噪声带来的训练不稳定
- EMA 能显著提升采样质量与稳定性
