# twodgen 问题清单（按严重程度排序）



## P0（阻断级 / Must-fix now）
- geometry 约束根本没生效：`train_metrics/config.jsonl:36-264` 显示 `geometry_config.use_geometry_fields=false`，`train_metrics/train_metrics.jsonl:3-20` 的 `loss_angle/loss_cond` 终究是 0，`pred_angle_out_rate` 挂 0，但 `train_metrics/tier0_metric.jsonl:1-27` 仍然在 valid_rate_eval=0.11、177 次 collision；根源是 `prepare_c2db_tokens.py:498-514` 彻底把缓存的 `coord_frame` 写成 `"raw"`，即便 `f_canon/gram6_canon/uv_angle` 都在，`train_tokens.py:1212-1233` 仍因 `coord_frame_meta != 'canon'` 把 `use_geometry_fields` 关掉。也就是说 geometry heads 连入口都没打开，导致新增的角度/condition loss 无法约束结构，collisions 依旧。需要让预处理在生成 canonical 缓存时把 `coord_frame` 标记为 `"canon"` 或修改 gating 逻辑，使得 `coord_frame_actual` 为 canonical 时 geometry 仍能运行。

## P1（高优先级）
- 2D 真空维度约束缺失：`train_metrics/tier1_2d_metrics.jsonl` 显示 `valid_2d_rate=0.045`、`cross_vacuum_rate=0.81`，且 `vacuum.mean=1.486` 远低于 `vacuum_min=15` 的目标；同时 `train_metrics/config.jsonl` 中 `vacuum_loss_weight=0.0`，导致模型没有学习保持足够真空区，2D 评估大幅失效。
- 晶格体积/角度分布塌缩：`train_metrics/tier0_metric.jsonl` 的 `volume.p10/p90` 全等，`per_sanmple.jsonl` 中 200 个样本体积几乎常数（~71.220），角度也集中在 90 度附近，说明 lattice/Gram 预测或条件应用被固定，采样无法反映真实尺度分布。

## P2（中优先级）
- 训练统计异常：`train_metrics/train_metrics.jsonl` 在 step 9900 出现 `min_dist_mean=Infinity`，同时 `collision_rate` 波动较大，可能存在 NaN/Inf 结构或 min_dist 计算回退到默认值，训练/日志需要增加异常捕获与样本溯源。

## P3（低优先级）
- 评估链路依赖缺失/占位：`property_predict.py` 仍为 mock（`twodgen/evaluate/property_predict.py:1-80`），形成能评估依赖的 `ref_energies.json` 无生成路径（`twodgen/evaluate/formation_energy.py:30-60`），影响评估完整性但不直接阻断训练。
