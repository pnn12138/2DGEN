
# 待做任务清单

- [P0] 修复 `prepare_c2db_tokens.py` 与缓存元数据，使具备 canonical `f_canon/gram6_canon` 的样本同时写入 `coord_frame="canon"`（或增加 gating 逻辑）让训练能够在 geometry 头上真正激活 `uv_angle/z_norm/lattice_param/t`，并确认 `train_metrics/config.jsonl` 里 `geometry_config.use_geometry_fields=true`、`loss_angle/loss_cond` 有值。
- [P0] 重新跑训练+采样评估（使用上面的 split + canonical cache）以验证 valid_rate/ collision 出口改善，结果按要求记入 `twodgen/history.md`，并把问题从 `twodgen/problem.md` 移除。
- [P1] 检查 geometry loss 量级：确认 `noise_scale_zn` 或其他 z_norm 相关超参的默认值已经协调，必要时在训练配置里提供更低的默认或可控 CLI（记录在 `twodgen/process.md`），并复跑小批量以验证 `loss_zn` 不再压倒其他损失。
- [P2] 将 `c2db_quality.jsonl` 的质量桶/硬通行控制接入 `train_tokens.py` 的默认数据流（如果存在 quality json 就先过滤），并在采样/评估脚本里复用同一份 filtering 结果，确保训练与评估实际勾选了质量标签。
- [P1] 开启或补齐 vacuum 约束：给 `train_tokens.py` 默认配置加入 `vacuum_loss_weight>0` 与合理 `vacuum_min`，采样/评估时也显式传入 `vacuum_min`，目标是降低 `cross_vacuum_rate` 并提升 `valid_2d_rate`（记录到 `twodgen/process.md`）。
- [P1] 追踪晶格分布塌缩：检查采样时是否覆盖预测的 Gram/lattice（避免被 `cell_init` 或条件常量覆盖），确认 `volume` 与角度分布有方差；必要时加入 lattice 统计可视化并复跑验证。
- [P2] 针对 `min_dist_mean=Infinity` 增加异常保护：在训练/统计中检测 NaN/Inf min_dist 与空邻接，记录样本 id 与来源，确保训练日志与 eval 指标不被污染。
