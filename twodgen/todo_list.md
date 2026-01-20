
# 待做任务清单

- [P0] 复核训练日志：用带 canonical 的 npz 重新训练后确认 `train_metrics.jsonl` 中 `geometry_config.use_geometry_fields=true`，并且 `loss_angle/loss_cond` 有非空记录。
- [P0] 重新跑训练+采样评估（使用上面的 split + canonical cache）以验证 valid_rate/ collision 出口改善，结果按要求记入 `twodgen/history.md`，并把问题从 `twodgen/problem.md` 移除。
- [P1] 开启 2D vacuum 约束：已配置默认值与评估入口参数，需通过采样/评估对比验证 `cross_vacuum_rate` 与 `valid_2d_rate` 的改善。
- [P1] 几何约束与碰撞治理加强：默认值已提升，仍需跑训练/采样验证 `tier0_metrics.json` 的 `valid_rate_eval`/`collision` 改善。
- [P1] 几何字段规范化与尺度对齐：已记录 `z_norm` 统计，需结合训练曲线验证 `loss_zn` 量级是否压制其它 loss。
- [P1] 真空/跨真空惩罚落地：默认已启用，仍需验证 `cross_vacuum_rate` 指标下降。
- [P1] 追踪晶格分布塌缩：采样侧已输出统计，仍需用新模型验证 lattice/Gram 是否摆脱“贴边”。
- [P2] 修复训练曲线写盘：代码已按 step 写入 `train_metrics.jsonl`，需跑训练验证产物落盘。
