## 可能的冗余/待确认点
- Legacy 网格路线已移除，无需再维护 `grid_to_structure()` 相关逻辑。
- `cond_lattice_mean/cond_lattice_std/cond_t_mean/cond_t_std` 在默认“仅化学式条件”下不会使用，可视情况延迟写入。

## 疑似冗余/危险代码（建议尽快处理）
- `data/dataset.py`：`CrystDataset` 未初始化 `self.data_list`，运行必炸；当前也没有被任何地方引用，建议删除或补全并接入。
- `common/crystal.py:clip_lattice()` 与 `model/atom_denoiser.py` 的 `v_min/v_max/cond_max`：目前采样/训练流程未实际调用晶胞 clip（只在评估中做 volume 判定），可能导致极端晶胞输出难以约束；建议要么接入采样投影/clip，要么移除这些未使用参数以免误导。

## 结构性重复（可以合并）
- `scrip/eval_samples.py`、`scrip/plot_compare.py`、`scrip/plot_eval.py`：仅是 `evaluate/*` 的 import wrapper；保留也可以，但建议统一入口（比如只留 `scrip/*` 作为 CLI）。
