# twodgen 问题清单（按严重程度排序）



## P0（阻断级 / Must-fix now）
- 训练目标与评估口径严重不一致：`train_metrics/tier0_metric.jsonl:1` 的 valid_rate_eval=0、angle_out_of_range_rate=1.0，说明所有样本在角度判据上失效；训练 loss 仅对 `loss_g` 做 MSE 拟合，没有任何角度/condition 约束，因此训练 loss 下降并不能改善 valid_rate，评估结论失真（`twodgen/evaluate/eval_samples.py:326`）。已在训练 loss 中新增 angle/cond 约束与监控，但需复跑确认有效率修复。
- t 归一化存在潜在数据泄漏：`slab_t` 的归一化优先读 npz 中 `cond_t_mean/cond_t_std`（`twodgen/scrip/train_tokens.py:1170-1180`），而它们通常由全量数据生成（`twodgen/data/prepare_c2db_tokens.py:540-541`）。当使用 `--split-json` 时，相当于用全数据统计量标准化训练子集，满足“标准化用到了全数据统计量”的 P0 判据。已改为 split-only 统计，但需复跑验证。

## P1（高优先级）
- geometry loss 量级失衡：`train_metrics/train_metrics.jsonl` 中 `loss_zn` 常年 100~200，而 `loss_f/loss_g/loss_z` 基本 <5；已下调 `noise_scale_zn` 默认值并加入 CLI 控制（`twodgen/common/atom_diffusion.py:29`、`twodgen/scrip/train_tokens.py:149`），但仍需复跑验证是否缓解。
- coord_frame 回退时仍可能混用 canonical geometry：当 `coord_frame_actual` 回退到 `"raw"`（`twodgen/data/c2db_dataset.py:333`），但 `uv_angle/z_norm/lattice_param` 仍存在时，geometry heads 会根据字段存在与否被启用（`twodgen/scrip/train_tokens.py:1140-1142`）；已加入 `coord_frame` 元数据一致性检查并在不一致时禁用几何头，但需要复跑确认。

## P2（中优先级）
- 清洗/标签产物未接入训练：`c2db_quality.jsonl` 等标注未进入 `train_tokens` 的默认流程（`rg -n c2db_quality` 仅见文档），导致数据质量控制无法作用于实际训练与评估，影响效率与可维护性。

## P3（低优先级）
- 评估链路依赖缺失/占位：`property_predict.py` 仍为 mock（`twodgen/evaluate/property_predict.py:1-80`），形成能评估依赖的 `ref_energies.json` 无生成路径（`twodgen/evaluate/formation_energy.py:30-60`），影响评估完整性但不直接阻断训练。
