
# 待做任务清单

- [P0] 修复 `prepare_c2db_tokens.py` 与缓存元数据，使具备 canonical `f_canon/gram6_canon` 的样本同时写入 `coord_frame="canon"`（或增加 gating 逻辑）让训练能够在 geometry 头上真正激活 `uv_angle/z_norm/lattice_param/t`，并确认 `train_metrics/config.jsonl` 里 `geometry_config.use_geometry_fields=true`、`loss_angle/loss_cond` 有值。
- [P0] 重新跑训练+采样评估（使用上面的 split + canonical cache）以验证 valid_rate/ collision 出口改善，结果按要求记入 `twodgen/history.md`，并把问题从 `twodgen/problem.md` 移除。
- [P1] 开启 2D vacuum 约束：`twodgen/scrip/train_tokens.py` 的 `--vacuum-loss-weight` 默认是 0.0（未启用），需要设为正值并写入 README/默认配置；同时在 `twodgen/scrip/sample_tokens.py` 与 `twodgen/evaluate/eval_samples.py` 的评估入口显式传入 `--vacuum-min`，用采样/评估对比验证 `cross_vacuum_rate` 与 `valid_2d_rate` 是否下降/提升。
- [P1] 几何约束与碰撞治理加强：明确提升 `min_dist` 惩罚与 repulsion 强度（训练侧 `min_dist_train_weight`、采样侧 `min_dist_iter/min_dist_strength`），对齐 `eval_min_dist`，并以 `tier0_metrics.json` 的 `valid_rate_eval`/`collision` 为验收指标。
- [P1] 几何字段规范化与尺度对齐：核查 `z_norm` 的噪声尺度与归一化逻辑（`twodgen/scrip/train_tokens.py` + `twodgen/data/preprocess.py`），确保 `loss_zn` 不压制其他 loss；必要时添加更细粒度 CLI 以调参。
- [P1] 真空/跨真空惩罚落地：训练侧启用 `vacuum_loss_weight` 并联动 `vacuum_min`；采样/评估端强制记录 `vacuum_min`，并将 `cross_vacuum_rate` 作为 T2.3 验收指标。
- [P1] 追踪晶格分布塌缩：定位 `twodgen/scrip/sample_tokens.py` 的 conditioning 路径与 `twodgen/model/atom_denoiser.py` 的 `cell_init` 行为，确认 lattice/Gram 是否被默认初始化或条件常量覆盖；补充统计（volume/angles 分布、Gram6 方差）或调试输出，验证模型是否真正学习并更新 lattice。
- [P1] 晶格多样性增强：在不破坏物理约束前提下，增加 lattice 相关随机性或扰动（例如 `cell_init_noise` 或 `lattice_param` 采样策略），并对 volume/angle 分布方差设置量化门槛。
- [P2] 训练统计异常定位：在 `twodgen/scrip/train_tokens.py` 的 min_dist 日志中，对 `n_atoms<2` 或全 inf 的 batch 做过滤/计数，记录样本索引以追踪 `min_dist_mean=Infinity` 根因（空邻接/掩码异常）。
- [P2] 评估 NaN 字段处理：`twodgen/evaluate/eval_samples.py` 中 `min_dist_same_elem` 在无重复元素时写入 NaN，需决定是否改为 null/空并记录计数，避免下游脚本误读。
- [P2] 评估输出一致性：统一 `per_sample.jsonl` 的命名与字段（修正 `per_sanmple.jsonl`），确保 `plot_eval/merge_reports` 等下游脚本可直接消费。
- [P2] 将 `c2db_quality.jsonl` 的质量桶/硬通行控制接入 `train_tokens.py` 的默认数据流（如果存在 quality json 就先过滤），并在采样/评估脚本里复用同一份 filtering 结果，确保训练与评估实际勾选了质量标签。
