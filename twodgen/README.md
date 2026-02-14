# twodgen 子项目说明

## 内容概览
- **默认路线（token）**：`model/atom_transformer.py` + `model/atom_denoiser.py`，以 `(Z,F,g)` token 表示进行扩散训练与采样。
- **扩散与损失**：`common/atom_diffusion.py`（连续变量预测 x0，训练时换算为 v；动态权重 + 可选 Flow-Matching）。
- **数据**：`data/prepare_c2db_tokens.py` 生成 token 缓存 npz；`data/c2db_dataset.py` 提供 `C2DBTokenNPZDataset`。
- **测试/训练**：`scrip/test_tokens.py`/`scrip/train_tokens.py`/`scrip/sample_tokens.py` 为 token 路线。

## 当前对齐状态
- A++ v3 预处理字段已写入 npz（`uv_angle` 等）。
- 邻居图支持 slab 2D PBC（默认 `--pbc-mask 1,1,0`），训练/采样均在线构图。
- 采样默认输出 Tier‑0/1 评估结果（写入 `out-dir/eval/`）。
- 训练/采样默认使用 canonical 坐标系，并通过 `order_idx` 对齐 per-atom 字段。
- 评估脚本去除了采样循环依赖；`plot_compare` 默认使用精确 MIC。

## 快速开始（Token 路线）
1) 预处理（生成 token npz）：
```bash
uv run python -m twodgen.data.prepare_c2db_tokens \
  --csv data/C2DB/c2db_summary.csv \
  --out data/C2DB/cache/c2db_tokens_2d_based.npz \
  --max-atoms 24 \
  --g-scale 100 \
  --pbc-mask 1,1,0 \
  --vacuum-min 15.0 \
  --bond-cut 3.0
```
说明：
- `lattice` 为行向量基矢，`cart = frac @ lattice`；Gram6 为 `G = lattice @ lattice^T`。
- 旧版 `.npz` 缺少 `gram6_convention` 时请重新预处理生成新缓存。
- 生成 canonical 字段后会自动把 `coord_frame=canon` 写入 metadata，有条件重新跑一次预处理即可让 downstream 确认 geometry heads 可用。
- NumPy >= 2.0 环境下请使用最新代码生成缓存（历史版本曾因 `np.unicode_` 被移除导致解析全失败，表现为 saved 0 samples）。

2) 生成 train/heldout 划分（供训练/采样使用）：
```bash
uv run python -m twodgen.data.create_c2db_split \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out data/C2DB/cache/c2db_tokens_split.json \
  --heldout-fraction 0.1 \
  --t-bins 10
```
说明：
- 若 `--npz` 是空缓存（0 samples），请先回到步骤 1 重新生成 token cache 再创建 split。

3) 训练（默认使用 train split）：
```bash
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --split-json data/C2DB/cache/c2db_tokens_split.json \
  --split train \
  --model-size base \
  --max-steps 20000 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-workers 4 \
  --tb-logdir outputs/tb_runs/train_tokens \
  --tb-interval 200 \
  --alert-steps 10000 \
  --save-dir outputs/checkpoints \
  --seed 0
```
说明：
- 默认开启 collision curriculum、`--filter-min-dist-below 1.35` 与 `--min-dist-train-weight 0.08`，显式减少重叠样本。  
- 训练默认启用 `--vacuum-loss-weight 0.1`（若要关闭需显式设为 0），与评估端的 `--vacuum-min 15` 对齐以强化 2D 真空约束。
- 可选启用 cross-vacuum 训练惩罚：`--cross-vacuum-loss-weight`/`--cross-vacuum-bond-cut`（默认 0 代表关闭）。
- 若要验证诊断仪表板与告警，建议使用 `--tb-logdir` + `--tb-interval`，并保留默认 `--alert-*` 参数（前 1 万步输出异常告警）。
- 默认启用 `--auto-volume-bounds`，会按训练 split 的体积分布 p1/p99 自动设置 `volume_min/volume_max`，避免体积阈值漏设。
- `cell_rep=cholesky6` 时会从训练 split 的 lattice 统计按维度的 `chol_log_min_vec/max_vec` 并写入 checkpoint；采样/训练的 clamp 与相关 loss 会优先使用按维 bound（旧 checkpoint 会回退到标量 `chol_log_min/max`）。
- 若前置质量筛选已产出 `data/C2DB/clean/c2db_quality.jsonl`，可用 `--quality-jsonl`/`--quality-buckets`/`--quality-hard-pass-only` 只取 `good`/`risk`、`hard_pass` 行；clean 脚本详见 `twodgen/data/clean_c2db_2d.py`。

4) 采样（默认对 heldout 条件生成）：
```bash
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 200 \
  --steps 50 \
  --method heun \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out-dir outputs/samples_tokens \
  --seed 0 \
  --cond-split-json data/C2DB/cache/c2db_tokens_split.json \
  --cond-split heldout
```
说明：
- 如果训练/采样要启用 geometry/project-geometry，请先用最新的 `.npz` 重新预处理（含 canonical 字段）并重新训练，避免旧 `coord_frame=raw` metadata 使 geometry heads 被关闭。
- 如果要启用（或验证）按维度的 `chol_log_min_vec/max_vec` 防坍缩 clamp，需要使用包含该字段的新 checkpoint（用当前代码重新训练一次即可写入）。
```

## 评估（samples.npz）
```bash
# 正常评估（输出 per_sample.jsonl 与 Tier-0/1）
uv run python -m twodgen.evaluate.eval_samples \
  --samples outputs/samples_tokens/samples.npz \
  --out-dir outputs/samples_tokens/eval

# MIC 自检（不依赖 samples.npz）
uv run python -m twodgen.evaluate.eval_samples --self-check
```

## 快速自测
Token 路线：
```bash
uv run python -m twodgen.scrip.test_tokens
```

### Baseline 评估（`eval_run_001`）
为保证可复现与 train/held-out 指标显式区分，推荐使用：
- split：`uv run python -m twodgen.data.create_c2db_split ...`
- 统一评估入口：`uv run python -m twodgen.evaluate.eval_run_001 ...`

详情与 `valid_rate` 定义见 `twodgen/baselines/eval_run_001.md`。

评估方法举例：
- 基础评估（输出 `per_sample.jsonl` 与 Tier‑0/1 结果）：
  ```bash
  uv run python -m twodgen.evaluate.eval_samples \
    --samples outputs/samples_tokens/samples.npz \
    --out-dir outputs/samples_tokens/eval
  ```
- 结合数据集统计约束体积范围（p1/p99）：
  ```bash
  uv run python -m twodgen.evaluate.eval_samples \
    --samples outputs/samples_tokens/samples.npz \
    --stats-npz data/C2DB/cache/c2db_tokens_2d_based.npz
  ```
- 生成评估分布图（直方图 + thickness/vacuum 散点）：
  ```bash
  uv run python -m twodgen.evaluate.plot_eval \
    --per-sample outputs/samples_tokens/eval/per_sample.jsonl \
    --out-dir outputs/samples_tokens/eval/plots
  ```
- 与真实数据分布对比（半小提琴图）：
  ```bash
  uv run python -m twodgen.evaluate.plot_compare \
    --samples outputs/samples_tokens/samples.npz \
    --dataset data/C2DB/cache/c2db_tokens_2d_based.npz \
    --coord-frame canon --mic-mode exact \
    --out outputs/samples_tokens/eval/compare_violin.png
  ```

## CIF 评估流水线（Phase 3）
```bash
uv run python -m twodgen.evaluate.run_pipeline \
  --cif-dir data/CIF_INPUTS \
  --out-dir outputs/cif_eval \
  --ref-energies data/ref_energies.json \
  --vacuum-min 15.0 \
  --vacuum-ratio-min 3.0 \
  --target-elements C,O \
  --formation-max 0.0
```

## Tier-2 性质预测（Phase 3）
默认使用 `twodgen.evaluate.property_predict` 的启发式模型（vacuum/thickness/min_dist 线性组合 + cross-vacuum penalty + valid bonus）来填充 Tier-2 指标：
```bash
uv run python -m twodgen.evaluate.property_predict \
  --per-sample outputs/samples_tokens/eval/per_sample.jsonl \
  --out-dir outputs/samples_tokens/eval/property \
  --property-key band_gap
```
- 如果希望调整预测，可以通过 `--vacuum-weight`/`--thickness-weight`/`--min-dist-weight`/`--cross-vacuum-penalty`/`--valid-bonus` 控制加权；`--mode` 支持 `heuristic`（默认）、`constant` 与 `random`，`--mock-predict` 仍强制随机。  
- 预测结果会写入 `per_sample_property.jsonl` 与 `property_metrics.json`，可串接到 `twodgen/evaluate/merge_reports.py` 做后续合并。

## 模型规模切换
模型规模配置集中在 `twodgen/model/model_sizes.py`，默认使用 `base`。

切换到更小模型（例如 tiny）：
```bash
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --model-size tiny \
  --epochs 50 --batch-size 256 --lr 1e-4
```

切换到更大模型（例如 xl）：
```bash
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --model-size xl \
  --epochs 100 --batch-size 128 --lr 5e-5
```
自定义规模请在 `twodgen/model/model_sizes.py` 中添加条目后使用 `--model-size` 选择。
