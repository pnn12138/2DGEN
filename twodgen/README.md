# twodgen 子项目说明

## 内容概览
- **默认路线（token）**：`model/atom_transformer.py` + `model/atom_denoiser.py`，以 `(Z,F,g)` token 表示进行扩散训练与采样。
- **扩散与损失**：`common/atom_diffusion.py`（连续变量预测 x0，训练时换算为 v；动态权重 + 可选 Flow-Matching）。
- **数据**：`data/prepare_c2db_tokens.py` 生成 token 缓存 npz；`data/c2db_dataset.py` 提供 `C2DBTokenNPZDataset` 与 `C2DBAtomDataset`。
- **测试/训练**：`scrip/test_tokens.py`/`scrip/train_tokens.py`/`scrip/sample_tokens.py` 为 token 路线。

## 当前对齐状态
- A++ v3 预处理字段已写入 npz（`uv_angle` 等）。
- 邻居图支持 slab 2D PBC（默认 `--pbc-mask 1,1,0`），训练/采样均在线构图。
- 采样默认输出 Tier‑0/1 评估结果（写入 `out-dir/eval/`）。
- 训练/采样默认使用 canonical 坐标系，并通过 `order_idx` 对齐 per-atom 字段。
- 评估脚本去除了采样循环依赖；`plot_compare` 默认使用精确 MIC。

## 数据预处理（Token 默认）
1. 准备原始 CSV：`data/C2DB/c2db_summary.csv`（已包含 CIF 文本）。
2. 生成 token 缓存 npz（默认启用 A++ v3 预处理并写入 canonical slab 特征）：
  ```bash
  uv run python -m twodgen.data.prepare_c2db_tokens \
    --csv data/C2DB/c2db_summary.csv \
    --out data/C2DB/cache/c2db_tokens_2d_based.npz \
    --max-atoms 24 \
    --g-scale 100
  ```
  - 约定：`lattice` 为“行向量基矢”，笛卡尔坐标满足 `cart = frac @ lattice`；对应 Gram6 使用 `G = lattice @ lattice^T`（并写入 `gram6_convention=row_lattice`）。
  - 旧版 `.npz` 缺少 `gram6_convention` 时需重新预处理或迁移：
    `uv run python -m twodgen.data.migrate_gram6_convention --in <old.npz> --out <new.npz>`
  - `--max-atoms`：最多保留的原子数（超出则跳过该行）。
  - `--g-scale`：Gram6 缩放（训练中会乘回去恢复晶胞）。
  - `--niggli-reduce`：对晶胞做 Niggli 规约（可选，较慢）。
  - `--preprocess-v3/--no-preprocess-v3`：写入 A++ v3 预处理字段（默认启用）。
3. 训练（邻居基于扩散状态在线构建）：
  ```bash
  uv run python -m twodgen.scrip.train_tokens \
    --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-4 \
    --model-size base \
    --seed 0 \
    --save-dir outputs/checkpoints
  ```
  - 完整评估需完成至少一轮完整训练（例如 100 epochs）；短跑仅用于链路健康检查，不具备可比的有效率指标。
  也可用 CLI：`twodgen-train --npz ...`
  - 训练脚本已固定为推荐默认值（`coord_frame=canon`、`align_atoms=True`、`cell_rep=cholesky6`、`cell_init=iso`、`use_geometry_fields=True`、`use_condition=True`）。
  - 条件字段默认 `counts_vector,lattice_param,t`，并对 `lattice_param,t` 做归一化。

## Token 版扩散（默认）
基于 `(Z, F, g)` token 表示的 Transformer 扩散模型（默认路线）：

训练：
```bash
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --epochs 100 --batch-size 256 --lr 1e-4 \
  --model-size base \
  --seed 0 \
  --save-dir outputs/checkpoints
```
脚本其余参数已固定为推荐默认值；如需调整请直接修改脚本中的默认设置。

采样与导出：
```bash
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 10 --steps 50 --method heun \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out-dir outputs/samples_tokens \
  --seed 0
```
也可用 CLI：`twodgen-sample --checkpoint ...`
采样脚本已固定为推荐默认值（`coord_frame=canon`、`project_geometry=True`、`use_ema=True`、`min_dist_project=True` 等）。

推荐采样参数（更稳的晶胞先验）：
```bash
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 10 --steps 50 --method heun \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out-dir outputs/samples_tokens \
  --seed 0
```

## 网格版扩散（Legacy）
旧版网格路线已移除。

## 快速自测
Token 路线：
```bash
uv run python -m twodgen.scrip.test_tokens
```

评估（已有 samples.npz）：
```bash
uv run python -m twodgen.evaluate.eval_samples --samples outputs/samples_tokens/samples.npz
```
也可用 CLI：`twodgen-eval --samples ...`

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

## 端到端示例（预处理 → 训练 → 采样 → 评估）
```bash
# 1) 预处理
uv run python -m twodgen.data.prepare_c2db_tokens \
  --csv data/C2DB/c2db_summary.csv \
  --out data/C2DB/cache/c2db_tokens_2d_based.npz \
  --max-atoms 24 \
  --g-scale 100

# 2) 训练
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --model-size base \
  --seed 0 \
  --save-dir outputs/checkpoints

# 3) 采样
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 200 \
  --steps 50 \
  --method heun \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out-dir outputs/samples_tokens \
  --seed 0

# 4) 评估
uv run python -m twodgen.evaluate.eval_samples \
  --samples outputs/samples_tokens/samples.npz \
  --out-dir outputs/samples_tokens/eval
```

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
