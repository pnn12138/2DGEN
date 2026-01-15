# Phase 0 Baseline: `eval_run_001`

本文件用于“固化 baseline 实验配置”和“规范化评估输出格式”，确保结果可复现。

## 1) 显式划分 train / held-out（验证集）

先生成 token cache（见 `twodgen/README.md`），然后创建 split：

```bash
uv run python -m twodgen.data.create_c2db_split \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out data/C2DB/cache/splits/c2db_train_heldout_v1.json \
  --heldout-fraction 0.1 \
  --seed 0 \
  --t-bins 10
```

split 文件包含：
- `split.train_indices`：训练集（参与训练）
- `split.heldout_indices`：held-out 验证集（不参与训练）
- `distribution_checks.*`：`n_atoms / elements / t_bin` 的粗粒度分布差异检查（max abs diff）

## 2) 固化训练配置（baseline）

训练时显式指定使用 train split：

```bash
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --split-json data/C2DB/cache/splits/c2db_train_heldout_v1.json \
  --split train \
  --epochs 100 --batch-size 256 --lr 1e-4 \
  --model-size base \
  --seed 0 \
  --save-dir outputs/checkpoints
```

每次训练都会在 `outputs/checkpoints/<RUN_STAMP>/` 写入：
- `config.json`：完整训练配置 + `run_metadata`（含 git commit / argv / 版本）
- `train_metrics.jsonl`：训练过程指标（jsonl）
- `atomdenoiser_{last,best}.pt`：checkpoint

## 3) 固化评估（`eval_run_001`）

统一使用 **条件采样 2000 个**，并区分：
- condition reconstruction（训练集条件，`cond_split=train`）
- conditional generation（held-out 条件，`cond_split=heldout`）

```bash
uv run python -m twodgen.evaluate.eval_run_001 \
  --checkpoint outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --split-json data/C2DB/cache/splits/c2db_train_heldout_v1.json \
  --out-dir outputs/eval_run_001 \
  --num-samples 2000 \
  --steps 50 \
  --method heun \
  --seed 0
```

输出目录结构：
- `outputs/eval_run_001/report.json`：两类评估的汇总报告（含 tier0/tier1）
- `outputs/eval_run_001/condition_reconstruction_train/eval/`：训练集条件重构的 `per_sample.jsonl` / `tier0_metrics.json` / `tier1_2d_metrics.json`
- `outputs/eval_run_001/conditional_generation_heldout/eval/`：验证集条件生成的同结构输出

## 4) `valid_rate` 定义（以评估脚本为准）

baseline 使用 `tier0_metrics.json` 中的 `valid_rate_eval`：
- 定义：样本通过 `twodgen.evaluate.eval_samples` 的所有 validity 条件（`tier0_metrics.json:valid_criteria` 列出）
- 采样规模：`--num-samples 2000`
- 任务划分：train split（condition reconstruction）与 held-out split（conditional generation）分开统计

注意：`samples.npz` 中的 `valid_rate` 是采样导出阶段的快速检查（基于 min_dist/volume 等），不等价于 `valid_rate_eval`。

