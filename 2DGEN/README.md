# 2DGEN 子项目说明

## 内容概览
- **默认路线（token）**：`model/atom_transformer.py` + `model/atom_denoiser.py`，以 `(Z,F,g)` token 表示进行扩散训练与采样。
- **扩散与损失**：`common/atom_diffusion.py`（v-pred + 动态权重 + 可选 Flow-Matching）。
- **数据**：`data/prepare_c2db_tokens.py` 生成 token 缓存 npz；`data/c2db_dataset.py` 提供 `C2DBTokenNPZDataset` 与 `C2DBAtomDataset`。
- **测试/训练**：`scrip/test_tokens.py`/`scrip/train_tokens.py`/`scrip/sample_tokens.py` 为 token 路线。

## 当前对齐状态
- A++ v3 预处理字段已写入 npz（`uv_angle` 等）。
- 邻居图支持 slab 2D PBC（默认 `--pbc-mask 1,1,0`），可选缓存 `nbr_*` 并在训练中复用。

## 数据预处理（Token 默认）
1. 准备原始 CSV：`data/C2DB/c2db_summary.csv`（已包含 CIF 文本）。
2. 生成 token 缓存 npz（默认启用 A++ v3 预处理并写入 canonical slab 特征；此命令已包含邻居图缓存，无需额外步骤）：
  ```bash
  uv run python 2DGEN/data/prepare_c2db_tokens.py \
    --csv data/C2DB/c2db_summary.csv \
    --out data/C2DB/ache/c2db_tokens_2d_based.npz \
    --max-atoms 24 --g-scale 100 \
    --cache-neighbors --neighbor-k 16
  ```
  - `--max-atoms`：最多保留的原子数（超出则跳过该行）。
  - `--g-scale`：Gram6 缩放（训练中会乘回去恢复晶胞）。
  - `--niggli-reduce`：对晶胞做 Niggli 规约（可选，较慢）。
  - `--preprocess-v3/--no-preprocess-v3`：写入 A++ v3 预处理字段（默认启用）。
  - `--cache-neighbors`：在 npz 内缓存 slab kNN 邻居图（`nbr_*` 字段）。
  - `--neighbor-k`：缓存邻居图的 k（默认 16）。
3. 训练复用缓存邻居图（推荐 slab 场景）：
  ```bash
  uv run python 2DGEN/scrip/train_tokens.py \
    --npz data/C2DB/ache/c2db_tokens_2d_based.npz \
    --epochs 100 --batch-size 256 --lr 1e-4 \
    --weight-decay 1e-2 --betas 0.9,0.95 --warmup-steps 500 \
    --min-lr 1e-6 --lr-schedule cosine --clip-grad 1.0 --ema \
    --g-scale 100 --k-neighbors 32 \
    --cell-rep cholesky6 --pbc-mask 1,1,0 \
    --use-precomputed-neighbors \
    --use-condition
  ```
  - `--g-scale` 应与 npz 内 `g_scale` 一致（脚本会提示不一致告警）。
  - `--cell-rep cholesky6` 下 `chol_log_min/max` 会从数据统计自动估计（内部尺度：物理长度除以 `sqrt(g_scale)`），一般不需要手动覆盖；若你手动传 `--chol-log-min/max`，务必使用内部尺度单位。
  - 可用 `--model-size {tiny,base,large,xl}` 控制模型规模（默认 `base`）；也可用 `--embed-dim/--depth/--num-heads/...` 显式覆盖。

## Token 版扩散（默认）
基于 `(Z, F, g)` token 表示的 Transformer 扩散模型（默认路线）：

训练：
```bash
uv run python 2DGEN/scrip/train_tokens.py \
  --npz data/C2DB/ache/c2db_tokens.npz \
  --epochs 100 --batch-size 256 --lr 1e-4 \
  --weight-decay 1e-2 --betas 0.9,0.95 --warmup-steps 500 \
  --min-lr 1e-6 --lr-schedule cosine --clip-grad 1.0 --ema \
  --g-scale 100 --k-neighbors 32 \
  --cell-rep cholesky6 --pbc-mask 1,1,0
```
可选：
- `--bucket-batches`：按原子数分桶减少 padding。
- `--niggli-reduce`：CSV 直读时对晶胞做 Niggli 规约。
- `--cell-rep gram6`：切回 Gram6（默认）。
- `--cell-init iso`：Cholesky-6D 采样先验用各向同性 `y_iso`，自动估计尺度与 clamp（也可手动覆盖）。
- `--weight-decay`：权重衰减（推荐 1e-2，小数据可降到 1e-3）。
- `--betas`：AdamW betas（推荐 0.9,0.95）。
- `--warmup-steps`：warmup 步数（按总 steps 的 5% 取值）。
- `--min-lr`：cosine 最小学习率（推荐 1e-6 或 lr*1e-2）。
- `--lr-schedule`：`cosine` 或 `constant`。
- `--clip-grad`：梯度裁剪（推荐 1.0）。
- `--ema` / `--ema-decay`：启用 EMA 与 decay（推荐 0.9999）。
- `--use-condition`：启用条件扩散（默认仅 `counts_vector`，即化学式计数；如需额外条件再用 `--cond-fields` 指定）。
- `--cond-fields`：自定义条件字段列表（例如 `counts_vector,lattice_param,t,xrd`）。当前默认不启用 XRD，仅预留接口。
- `--cond-normalize-fields`：需要做 z-score 的条件字段（默认 `lattice_param,t`）。
- `--use-precomputed-neighbors`：使用 npz 缓存的邻居图（`nbr_*`）替代在线 kNN。
- `--pbc-mask`：控制 MIC 的 PBC 维度，默认 `1,1,0`（仅面内周期，z 非周期）；3D 晶体可设 `1,1,1`。

采样与导出：
```bash
uv run python 2DGEN/scrip/sample_tokens.py \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 10 --steps 20 --method euler \
  --max-atoms 24 --g-scale 100 --npz data/C2DB/ache/c2db_tokens.npz \
  --use-ema \
  --pbc-mask 1,1,0 \
  --out-dir outputs/samples_tokens
```
可选：
- `--neighbor-update-steps`：采样时每 N 步更新 kNN（默认 1）。
- `--reduce-lattice` / `--niggli-reduce`：采样后晶胞规约。
- `--cell-init iso`：与训练一致时可开启各向同性先验。
- `--cond-npz`：条件扩散时提供包含 `counts_vector`/`lattice_param`/`t` 的 npz；可配合 `--cond-index` 或 `--cond-random`。
- `--project-each-step`：每步将 `frac/lattice` 投影回合法域（默认只在最终输出做投影）。
- `--use-ema`：若 checkpoint 中包含 EMA 权重则优先使用。
- `--pbc-mask`：控制 MIC 的 PBC 维度，默认 `1,1,0`（仅面内周期，z 非周期）；3D 晶体可设 `1,1,1`。

推荐采样参数（更稳的晶胞先验）：
```bash
uv run python 2DGEN/scrip/sample_tokens.py \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 10 --steps 50 --method heun \
  --max-atoms 24 --g-scale 100 --npz data/C2DB/ache/c2db_tokens_2d_based.npz \
  --use-ema \
  --cell-init iso \
  --pbc-mask 1,1,0 \
  --out-dir outputs/samples_tokens
```

## 网格版扩散（Legacy）
旧版网格路线已移除。

## 快速自测
Token 路线：
```bash
uv run python 2DGEN/scrip/test_tokens.py
```
