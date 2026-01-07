# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 本次“低合格率”直接结论（基于 `/home/pnn/2dgen/eval`）

现象（Tier-0/1 评估）：
- `valid_rate_eval = 0.238`，失败原因几乎全部是 `collision`（`fail_reason_counts.collision = 1524 / 2000`）
- `min_dist` 分布显著偏低：mean≈`1.187Å`，p10≈`0.651Å`；评估阈值是 `--eval-min-dist 1.5`
- 2D slab 指标（thickness/vacuum/cross_vacuum）基本正常，说明主要问题集中在**原子-原子近距离重叠**而非 slab 真空/厚度。

一句话：当前低合格率的主因不是“元素不对/厚度不对/晶格不对”，而是采样出来的结构里大量原子对的 MIC 距离 < `1.5Å`，被评估脚本判定为 collision。

## 进一步问题分析（基于当前代码实现）

### 1) PBC mask 不一致会扭曲最小距离
- 采样合法性检查与评估都依赖 `pbc_mask`。默认取自 checkpoint 的 `model_cfg.pbc_mask`，若旧 ckpt 缺失字段则回退到 `(1,1,0)`。
- 若训练/采样/评估之间 PBC mask 不一致（例如 slab 却被当成 3D），会导致 MIC 距离被低估，碰撞率异常升高。

### 2) coord-frame 与数据字段一致性需确认
- `--coord-frame canon` 需要 `npz` 内存在 `f_canon/lattice_canon`；若数据缺失则会回退到 raw 坐标，等价于“采样/评估假设与数据字段不一致”。
- 这种不一致会带来晶格尺度/分布偏移，间接影响 min_dist 分布。

### 3) coord_frame 回退时几何字段仍保留 canonical，存在混用风险
- `C2DBTokenNPZDataset` 会在缺少 `f_canon/gram6_canon` 时把 `coord_frame_actual` 设为 `raw`，但仍会返回 `uv_angle/z_norm/lattice_param` 等 canonical 字段。
- 如果训练时启用 geometry heads（或条件里用 `lattice_param/t`），就可能出现 **raw F/gram6 + canonical geom** 的混合输入，几何场与主扩散坐标系不一致。

### 4) composition 编码的 CLI 可控性不足
- composition encoder 已落地，但 `train_tokens.py` 未暴露 `--use-comp-encoder/--comp-embed-dim/--comp-pool-mode/--comp-use-frac` 等参数，导致只能改脚本默认值，消融不便。

## 训练侧问题分析（基于 `/home/pnn/2dgen/_train_/config.json` + `train_metric.jsonl`）

### 1) 训练/采样依赖一致性强，但未做一致性检查
- 训练使用 `coord_frame=canon`、`use_geometry_fields=true`、`pbc_mask=1,1,0`，采样必须严格对齐。
- 当前流程没有自动检查这些关键配置是否在采样时被覆盖或丢失（例如 `pbc_mask`、`coord_frame`）。

### 2) 训练指标表现正常但不代表结构质量
- loss 从约 `27.8` 降至 `~6.2`，说明优化收敛；但这只反映“噪声回归误差”，与结构合法性关联弱。
- 因此“训练看起来正常、采样却不合法”是该设定下的合理结果，而非纯粹的训练失败。

## 需要后续确认/排查的点（建议按优先级）
1. 采样的最小距离 repulsion 是否生效（默认开启，确认 `min_dist` 分布是否抬升）。
2. checkpoint 内的 `diffusion.cell_init` / `cell_rep` / `g_scale` 是否与训练配置一致。
3. 训练/采样使用的 `npz` 是否包含 `f_canon/lattice_canon`，并与默认 `coord_frame=canon` 对齐。
4. 若 `coord_frame_actual` 回退为 raw，是否自动禁用 geometry heads 或强制使用 raw 对齐的几何字段。

## 近期改动备注（影响复现）
- 模型已切换为**预测 x0**（训练时换算为 v），旧 checkpoint 与当前代码不兼容，需要重新训练。
- CLI 参数已精简为少量核心项，其他超参数固定为推荐默认值；如需修改请直接改脚本默认值。

## 推荐采样命令（对齐评估阈值 + 防碰撞后处理）
```bash
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 2000 --steps 50 --method heun \
  --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --out-dir outputs/samples_tokens
```
