# twodgen 问题清单（按严重程度排序）



## P1（高优先级）
- 2D 真空维度约束缺失：`train_metrics/tier1_2d_metrics.jsonl` 显示 `valid_2d_rate=0.045`、`cross_vacuum_rate=0.81`，且 `vacuum.mean=1.486` 远低于 `vacuum_min=15` 的目标；同时 `train_metrics/config.jsonl` 中 `vacuum_loss_weight=0.0`，导致模型没有学习保持足够真空区，2D 评估大幅失效。
- 晶格体积/角度分布塌缩：`train_metrics/tier0_metric.jsonl` 的 `volume.p10/p90` 全等，`per_sanmple.jsonl` 中 200 个样本体积几乎常数（~71.220），角度也集中在 90 度附近，说明 lattice/Gram 预测或条件应用被固定，采样无法反映真实尺度分布。

## P2（中优先级）
- 训练统计异常：`train_metrics/train_metrics.jsonl` 在 step 9900 出现 `min_dist_mean=Infinity`，同时 `collision_rate` 波动较大，可能存在 NaN/Inf 结构或 min_dist 计算回退到默认值，训练/日志需要增加异常捕获与样本溯源。

## P3（低优先级）
- 评估链路依赖缺失/占位：`property_predict.py` 仍为 mock（`twodgen/evaluate/property_predict.py:1-80`），形成能评估依赖的 `ref_energies.json` 无生成路径（`twodgen/evaluate/formation_energy.py:30-60`），影响评估完整性但不直接阻断训练。
