# Phase 2 落地验证方案

目标：验证已落地的 T2.1/T2.2/T2.3 功能在数据、训练、采样、评估闭环中工作正常，且指标方向符合预期。

## 1. 前置检查

### 1.1 代码与环境
- 环境：使用 `uv run ...` 运行所有命令
- 入口脚本：
  - 数据：`twodgen/data/prepare_c2db_tokens.py`
  - 训练：`twodgen/scrip/train_tokens.py`
  - 采样：`twodgen/scrip/sample_tokens.py`
  - 评估：`twodgen/evaluate/eval_samples.py`

### 1.2 数据缓存字段完整性
检查 `npz` 是否包含以下字段（用于 T2.1/T2.2/T2.3）：
- `counts_vector`
- `min_dist`, `collision_risk`, `min_dist_cut`, `min_dist_pbc_mask`
- `slab_thickness`, `slab_vacuum`, `low_vacuum_risk`, `cross_vacuum_bond`, `vacuum_min`

建议使用最小命令生成一次缓存（确保字段都存在）：
```
uv run python -m twodgen.data.prepare_c2db_tokens \
  --min-dist-cut 1.5 \
  --pbc-mask 1,1,0 \
  --vacuum-min 15.0
```

## 2. 训练验证

### 2.1 训练命令（最小验证）
建议先用小 epoch / 小 batch 做健康检查，确认损失和日志都正常：
```
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens.npz \
  --epochs 3 \
  --batch-size 64 \
  --min-dist-train-cut 1.5 \
  --min-dist-train-weight 0.02 \
  --comp-loss-weight 1.0 \
  --comp-loss-mode l1 \
  --vacuum-loss-weight 1.0 \
  --vacuum-min 15.0
```

如需验证 curriculum/过滤：
```
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens.npz \
  --epochs 5 \
  --curriculum-collision \
  --curriculum-epochs 2 \
  --curriculum-min-dist-cut 1.5 \
  --filter-min-dist-below 0.8
```

### 2.2 训练过程检查项
检查 `train_metrics.jsonl` 或控制台日志是否包含：
- `loss_min_dist`, `loss_comp`, `loss_vacuum`
- `collision_rate`, `min_dist_mean`, `min_dist_p10`
- （若启用不确定性加权）`s_comp`, `s_vacuum`

预期：
- `loss_comp`、`loss_vacuum` 在训练过程中逐步下降或稳定在可解释区间
- `collision_rate` 下行趋势，`min_dist_mean/p10` 上行趋势
- `loss_min_dist` 不为 0（可学习）

## 3. 采样与评估验证

### 3.1 采样命令
使用训练出的 checkpoint 进行采样，并开启 evaluation：
```
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint outputs/checkpoints/last.pt \
  --num-samples 100 \
  --steps 50 \
  --eval \
  --eval-min-dist 1.5 \
  --vacuum-min 15.0 \
  --reject-cross-vacuum
```

若要验证 composition 硬约束：
- 准备 `--cond-npz` 并在采样时使用 `--cond-npz`

### 3.2 采样输出检查项
检查 `samples.npz` 和 `sampling_config.json`：
- `min_dist_pre_mean` < `min_dist_post_mean`（repulsion 生效）
- `collision_pre` > `collision_post`
- `composition.hit_rate` 接近 1.0（若使用 `cond_counts_vector`）
- `vacuum`/`cross_vacuum_bond` 字段存在

### 3.3 评估输出检查项
检查 `eval/tier1_2d_metrics.json`：
- `valid_rate_eval` 提升（相对 baseline）
- `vacuum_ok_rate` 高
- `cross_vacuum_rate` 低
- `min_dist` 分布右移（`mean` / `p10` 上升）

## 4. 失败定位指南

### 4.1 训练期异常
- `loss_comp` 或 `loss_vacuum` 恒为 0：检查 `--comp-loss-weight` / `--vacuum-loss-weight`
- `counts_vector` 缺失：检查 `npz` 是否带 `counts_vector`，或 `cond_fields` 是否包含 `counts_vector`
- `min_dist` 缺失：检查 `npz` 中是否写入 `min_dist`

### 4.2 采样期异常
- composition 命中率低：确认采样是否传入 `--cond-npz` 且 counts sum == num_atoms
- vacuum 指标异常：检查采样时是否启用 `--vacuum-min` / `--reject-cross-vacuum`
- collision 过高：检查 `min_dist_repulsion` 参数、`min_dist_train_weight`

## 5. 验证结论模板

- 数据缓存字段完整性：是/否
- 训练损失可学习：是/否（`loss_min_dist/loss_comp/loss_vacuum`）
- collision 改善：是/否（`min_dist_post`、`collision_rate`）
- 组成命中：是/否（`composition.hit_rate`）
- vacuum 合格：是/否（`vacuum_ok_rate`、`cross_vacuum_rate`）

结论建议：若任一关键项失败，优先排查数据缓存字段与训练 CLI 参数，再检查 loss 权重与采样配置。
