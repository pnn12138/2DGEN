## 可能的冗余/待确认点
- Legacy 网格路线已移除，无需再维护 `grid_to_structure()` 相关逻辑。
- `prepare_c2db_tokens.py` 写入的 `nbr_dist` 目前训练未使用，仅保留 `nbr_idx/nbr_mask` 即可。
- `cond_lattice_mean/cond_lattice_std/cond_t_mean/cond_t_std` 在默认“仅化学式条件”下不会使用，可视情况延迟写入。
