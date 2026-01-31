# twodgen 问题清单（按严重程度排序）

## P0（最高优先级，严重影响项目结论）
- 已修复（2026-01-31）：训练侧碰撞惩罚无梯度。`min_dist_train_weight` 现在在 `AtomVelocityLoss` 内使用 `pred_x0_f/pred_x0_g` 计算，梯度链路指向模型输出。

## P1（高优先级）
- 已修复（2026-01-31）：对称性损失不可导。`_symmetry_residual_loss` 仅保留为指标记录，训练 loss 不再叠加该常量项，避免误导。
- 已修复（2026-01-31）：真空损失只约束晶胞不约束原子位置。`vacuum loss` 使用 `pred_x0_f` 计算最大真空间隙，梯度回传到原子坐标预测。
- 已修复（2026-01-31）：训练/采样在缺少 canonical 字段时会直接崩溃。`twodgen/data/c2db_dataset.py` 已补充 `import warnings`，确保 fallback 到 raw 坐标。

## P2（中优先级）
- 已修复（2026-01-31）：评估缓存易产生陈旧指标。`eval_cache.npz` 现在校验 `CACHE_VERSION`、`bond_cut`、`pbc_mask` 以及样本 `mtime/size`，不一致即重建缓存。
- 待优化：训练/评估核心几何变换存在明显性能瓶颈。`twodgen/common/crystal.py::gram6_to_lattice/gram6_to_cholesky6` 仍是逐样本 Python 循环，后续考虑批量化/torch.linalg 批处理改写。
- 训练时重复计算全量 `min_dist`：`_compute_dataset_min_dist` 只从 `dataset.extra` 读取缓存，但 `min_dist` 存在 `C2DBTokenNPZDataset.min_dist`，导致每次启动都重算（大数据集显著拖慢训练启动）。


## P3（低优先级）
- 已修复（2026-01-31）：`twodgen/evaluate/cache.py::build_eval_cache` 的 `pbc_mask` 参数未使用。现在与 eval_samples 行为对齐，仅在 `pbc_mask[c_idx]==0` 时触发 3D cross-vacuum 检测。
- 已核验（2026-01-31）：`twodgen/loss/schedule.py::_normalize_keys` 当前不存在于仓库（已清理），无需处理。
