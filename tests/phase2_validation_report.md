# Phase 2 落地验证报告

执行时间：本地验证（完整数据集 + README 流程）

## 1. 验证范围
- T2.1 几何可行性（collision）数据/训练/采样/评估链路
- T2.2 组成可控性（counts_vector + comp loss + 强约束采样）
- T2.3 二维真空合理性（vacuum loss + vacuum 评估）

## 2. 数据准备与字段校验

### 2.1 生成 token 缓存（README 文件名对齐）
```
uv run python -m twodgen.data.prepare_c2db_tokens \
  --csv data/C2DB/c2db_summary.csv \
  --out data/C2DB/cache/c2db_tokens_2d_based.npz \
  --max-atoms 24 \
  --g-scale 100 \
  --min-dist-cut 1.5 \
  --pbc-mask 1,1,0 \
  --vacuum-min 15.0
```
输出：`data/C2DB/cache/c2db_tokens_2d_based.npz`
样本量：16838

### 2.2 字段检查结果
目标字段均存在：
- `counts_vector`
- `min_dist`, `collision_risk`, `min_dist_cut`, `min_dist_pbc_mask`
- `slab_thickness`, `slab_vacuum`, `low_vacuum_risk`, `cross_vacuum_bond`, `vacuum_min`

## 3. 训练验证（完整数据集）

### 3.1 训练命令
```
uv run python -m twodgen.scrip.train_tokens \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --epochs 1 \
  --batch-size 128 \
  --min-dist-train-cut 1.5 \
  --min-dist-train-weight 0.02 \
  --comp-loss-weight 1.0 \
  --comp-loss-mode l1 \
  --vacuum-loss-weight 1.0 \
  --vacuum-min 15.0 \
  --save-dir outputs/checkpoints_phase2_full
```

### 3.2 训练关键指标（train_metrics.jsonl）
- `loss_min_dist`: 0.0102
- `loss_comp`: 1.9183
- `loss_vacuum`: 0.0926
- `collision_rate`: 0.0781
- `min_dist_mean`: 2.1831
- `min_dist_p10`: 1.6367

结论：
- collision/comp/vacuum loss 均非零且可反传，训练链路正常。

## 4. 采样与评估验证（完整数据集）

### 4.1 采样命令（50 样本）
```
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint outputs/checkpoints_phase2_full/20260118_120816/atomdenoiser_last.pt \
  --cond-npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --num-samples 50 \
  --steps 30 \
  --vacuum-min 15.0 \
  --reject-cross-vacuum \
  --out-dir outputs/samples_phase2_full
```

### 4.2 采样结果（sampling_config.json）
- composition 命中率：`hit_rate = 1.0`
- 采样有效率：`valid_rate_samples = 0.0`
- min_dist pre/post：`mean = 455.356 / 455.356`，`p10 = 56.154 / 56.154`

### 4.3 评估命令
```
uv run python -m twodgen.evaluate.eval_samples \
  --samples outputs/samples_phase2_full/samples.npz \
  --vacuum-min 15.0
```

### 4.4 评估结果（tier0/tier1）
- `vacuum_ok_rate`: 1.0
- `cross_vacuum_rate`: 0.0
- `valid_rate_eval`: 0.0
- `angle_out_of_range_rate`: 1.0
- 组成匹配：`cond_match.exact_match_rate = 1.0`

结论：
- 组成硬约束命中率达标（T2.2 OK）。
- vacuum 指标达标（T2.3 OK）。
- 总体有效率为 0，主要因晶格角度超出评估范围（不属于 Phase2 目标，但影响整体 valid_rate）。

## 5. 角度异常排查（增加 lattice clip）

### 5.1 采样命令（开启 `--project-each-step`）
```
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint outputs/checkpoints_phase2_full/20260118_120816/atomdenoiser_last.pt \
  --cond-npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --num-samples 50 \
  --steps 30 \
  --vacuum-min 15.0 \
  --reject-cross-vacuum \
  --project-each-step \
  --out-dir outputs/samples_phase2_full_proj
```

### 5.2 结果摘要
- `angle_out_of_range_rate`: 0.02（明显下降）
- `valid_rate_eval`: 0.10
- `vacuum_ok_rate`: 0.0
- `cross_vacuum_rate`: 0.88
- `collision_post`: 45/50

结论：
- 开启逐步投影后，角度异常显著缓解，但碰撞与跨真空问题明显增多，整体有效率仍偏低。
- 说明角度问题可通过 lattice clip 缓解，但需要与 collision/vacuum 约束联动调参。

## 6. 已修复问题

1) 训练入口报错（标量/非样本维元数据被索引）
- 修复：仅保留与样本维一致的 `extra` 字段
- 结果：README 缓存可直接训练

2) 采样期 `--reject-cross-vacuum` 崩溃
- 修复：正确解包 `_min_dist_and_shifts` 返回值，并处理标量距离输出

3) 采样期 eval 缺 `vacuum_min`
- 修复：`sample_tokens.py` 传入 `vacuum_min` 至 `_eval_samples` 和 `build_eval_params`

4) 新增采样参数
- `sample_tokens.py` 增加 `--project-each-step` 以启用晶格 clip 与分数坐标投影

## 7. 结论
- 完整数据集上，数据→训练→采样→评估链路可跑通。
- T2.1：训练期 collision loss 生效；采样期 min_dist 无碰撞但尺度偏大（需后续排查晶胞/坐标尺度）。
- T2.2：组成硬约束命中率 1.0。
- T2.3：vacuum 指标达标，cross-vacuum 为 0。
- `valid_rate_eval=0` 主要由晶格角度超界导致；开启 `--project-each-step` 可缓解角度异常，但会引入严重 collision/cross-vacuum 退化，需进一步联合约束。
