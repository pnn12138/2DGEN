# twodgen 问题清单（按严重程度排序）
## P2（中优先级）
- 待优化：训练/评估核心几何变换存在明显性能瓶颈。`twodgen/common/crystal.py::gram6_to_lattice/gram6_to_cholesky6` 仍是逐样本 Python 循环，后续考虑批量化/torch.linalg 批处理改写。
- `twodgen/data/c2db_dataset.py` 的 `extra` 读入逻辑仍只接受数值 (`dtype.kind in {"i","u","f","b"}`)，如果后续预处理把 `spacegroup_symbol` 或其它字符串放入 `.npz`，dataset 会直接跳过无法复用，需要改为保留原始对象或单独存 metadata。
