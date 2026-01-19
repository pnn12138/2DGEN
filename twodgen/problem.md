# twodgen 问题清单（按严重程度排序）



## P1（高优先级）
- 2D 真空维度约束缺失：`train_metrics/tier1_2d_metrics.jsonl` 显示 `valid_2d_rate=0.015`、`cross_vacuum_rate=0.87`，且 `vacuum.mean=1.563` 远低于 `vacuum_min=15` 的目标；同时 `train_metrics/config.jsonl` 中 `vacuum_loss_weight=0.0`，导致模型没有学习保持足够真空区，2D 评估大幅失效。现状：未解决。
- 晶格体积/角度分布塌缩：`train_metrics/tier0_metric.jsonl` 的 `volume.p10/p90` 全等，`per_sanmple.jsonl` 中样本体积几乎常数（~71.236），角度也集中在 90 度附近，说明 lattice/Gram 预测或条件应用被固定，采样无法反映真实尺度分布。现状：未解决。
- Tier‑0 有效率过低/碰撞严重：`train_metrics/tier0_metric.jsonl` 显示 `valid_rate_eval=0.1` 且 `collision=180/200`，`min_dist.mean=1.043` 明显低于 `min_dist_cut=1.5`，说明采样结果大量发生原子重叠，结构物理性不足。新增。

## P2（中优先级）
- 训练统计异常：`train_metrics/train_metrics.jsonl` 多次出现 `min_dist_mean=Infinity`（如 step 50/700/2000），同时 `collision_rate` 波动较大，可能存在 NaN/Inf 结构或 min_dist 计算回退到默认值，训练/日志需要增加异常捕获与样本溯源。现状：未解决。
- 评估样本存在 NaN 字段：`train_metrics/per_sanmple.jsonl` 中 `min_dist_same_elem` 出现 NaN（如 id=98），说明同元素最小距离统计在某些样本下未被正确处理，可能导致指标失真。新增。

## P3（低优先级）
- 评估链路依赖缺失/占位：`property_predict.py` 仍为 mock（`twodgen/evaluate/property_predict.py:1-80`），形成能评估依赖的 `ref_energies.json` 无生成路径（`twodgen/evaluate/formation_energy.py:30-60`），影响评估完整性但不直接阻断训练。现状：未解决。
- 指标输出文件命名疑似拼写错误：`train_metrics/per_sanmple.jsonl` 与 README 中的 `per_sample.jsonl` 不一致，可能导致后续脚本无法自动发现指标文件。新增。
