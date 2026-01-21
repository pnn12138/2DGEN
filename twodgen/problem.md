# twodgen 问题清单（按严重程度排序）



## P1（高优先级）
- 晶格体积塌缩到 `chol_log_min` 下界（根因级）：`train_metrics/tier0_metric.jsonl` 中 `volume.p10/p90≈71.2367` 几乎常数，且这一数值**严格等于** `train_metrics/config.jsonl` 的 `chol_log_min=-0.8805823`、`g_scale=100` 所对应的下界体积：
  - `side_min = exp(chol_log_min) * sqrt(g_scale) ≈ 4.1454 Å`
  - `volume_min = side_min^3 ≈ 71.2367 Å^3`
  - 这说明采样时的 cell（`cell_rep=cholesky6`）在 `twodgen/common/crystal.py` 的 clamp（`cholesky6_to_lattice/cholesky6_to_gram6`）处**长期触底**，导致 lattice 尺度/角度分布不可学习地“贴边”，进而把后续所有几何指标一起带崩。现状：未解决。
- 2D 真空维度约束失效：`train_metrics/tier1_2d_metrics.jsonl` 显示 `valid_2d_rate=0.01`、`cross_vacuum_rate=0.85`，且 `vacuum.mean=1.507` 远低于目标 `vacuum_min=15`；同时 `train_metrics/config.jsonl` 中 `vacuum_loss_weight=0.0/lambda_vacuum=0.0`，训练侧完全未优化真空。更关键的是：在“晶格触底”情况下，`c_len` 最大也只有约 `4.15 Å`，理论上就不可能达到 `15 Å` 的真空要求，因此该问题与上条强耦合。现状：未解决。
- Tier‑0 有效率过低/碰撞严重：`train_metrics/tier0_metric.jsonl` 显示 `valid_rate_eval=0.08` 且 `collision=184/200`，`min_dist.median=0.971 < min_dist_cut=1.5`。在 `volume` 被压到下界后，真实笛卡尔距离整体缩小，采样末尾的 `min_dist` repulsion 只能挪 `frac`，无法把 lattice 拉大，因此碰撞很难从根上缓解。现状：未解决。

## P2（中优先级）
（已核验）训练/评估产物已对齐：训练曲线按 step 写入 `outputs/checkpoints/<run>/train_metrics.jsonl`，评估写入 `tier0_metrics.json`、`tier1_2d_metrics.json`、`per_sample.jsonl`（见 `twodgen/history.md`）。本节问题已移除。

## P3（低优先级）
- 评估链路依赖缺失/占位：`property_predict.py` 仍为 mock（`twodgen/evaluate/property_predict.py:1-80`），形成能评估依赖的 `ref_energies.json` 无生成路径（`twodgen/evaluate/formation_energy.py:30-60`），影响评估完整性但不直接阻断训练。现状：未解决。
