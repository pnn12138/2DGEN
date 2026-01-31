# twodgen data utilities

本目录以离线 token 缓存为主（由 `prepare_c2db_tokens.py` 生成），训练/采样直接读取 `npz`。

## Token 预处理（A++ v3 可选）
文件：`twodgen/data/prepare_c2db_tokens.py`

- 默认保留原有字段：`z`、`f`（分数坐标）、`atom_mask`、`lattice`、`gram6`。
- 默认会缓存 canonical slab 表示（全部已 padding 到 `max_atoms`）：
  - `z_canon`：canonical 排序后的元素序列
  - `uvz`：`(u, v, z_norm)` 三元坐标
  - `uv_angle`：`(cos 2πu, sin 2πu, cos 2πv, sin 2πv)` 的 torus 嵌入
  - `u`, `v`, `z_norm`：单独存储便于检索
  - `t`、`a_hat`、`b_hat`、`n`、`lattice_param`、`counts_vector`、`order_idx`
- 同时写入 `preprocess_version=A++_v3` 与 `eps_area/eps_inv/round_prec/z_norm_clip` 供复现。

## Token 缓存读取
文件：`twodgen/data/c2db_dataset.py`

- `C2DBTokenNPZDataset` 读取 `prepare_c2db_tokens.py` 生成的 `npz`。
- 若 `npz` 内包含 A++ v3 字段（如 `counts_vector`、`lattice_param`、`t`），会自动加入样本字典，供条件扩散使用。

## Clean-2D 数据集治理（硬过滤 + 质量标签）
文件：`twodgen/data/clean_c2db_2d.py`

- 对 `data/C2DB/c2db_summary.csv` 逐行解析 CIF，计算 slab 2D 指标（n_atoms / vacuum / thickness / cross-vacuum bond / min_dist）。
- **硬过滤**（可配置）：原子数上限、真空层下限、跨真空成键（基于 3D MIC + shift）。
- 生成三类产物：审计表 `c2db_audit_2d.csv`、过滤后 `c2db_clean_2d.csv`、以及质量标签 `c2db_quality.jsonl` 与统计报告 `c2db_clean_report.json`。
