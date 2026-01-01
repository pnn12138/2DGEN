# 2DGEN data utilities

本目录包含在线读取 `C2DBDataset` 与离线 token 缓存（由 `prepare_c2db_tokens.py` 生成）。

## C2DBDataset（在线预处理）
文件：`2DGEN/data/c2db_dataset.py`

- 逐行读取 `data/C2DB/c2db_summary.csv`，解析 CIF，超出 `max_atoms` 可选丢弃。
- 生成 `atomic_numbers`、`frac_coords`、`lattice_matrix`，并用 `_pad_1d`/`_pad_2d` 填充到固定长度；`atom_mask` 标记真实原子。
- 样本结构：`{"atomic_numbers", "frac_coords", "atom_mask", "lattice_matrix", "counts_vector"}`（torch 张量），元数据存入 `C2DBMetadata`。
- `collate_fn` 简单堆叠；`get_metadata` 按索引取元信息。

## Token 预处理（A++ v3 可选）
文件：`2DGEN/data/prepare_c2db_tokens.py`

- 默认保留原有字段：`z`、`f`（分数坐标）、`atom_mask`、`lattice`、`gram6`。
- 开启 `--preprocess-v3` 后会额外缓存 canonical slab 表示（全部已 padding 到 `max_atoms`）：
  - `z_canon`：canonical 排序后的元素序列
  - `uvz`：`(u, v, z_norm)` 三元坐标
  - `uv_angle`：`(cos 2πu, sin 2πu, cos 2πv, sin 2πv)` 的 torus 嵌入
  - `u`, `v`, `z_norm`：单独存储便于检索
  - `t`、`a_hat`、`b_hat`、`n`、`lattice_param`、`counts_vector`、`order_idx`
- 同时写入 `preprocess_version=A++_v3` 与 `eps_area/eps_inv/round_prec/z_norm_clip` 供复现。

## Token 缓存读取
文件：`2DGEN/data/c2db_dataset.py`

- `C2DBTokenNPZDataset` 读取 `prepare_c2db_tokens.py` 生成的 `npz`。
- 若 `npz` 内包含 A++ v3 字段（如 `counts_vector`、`lattice_param`、`t`），会自动加入样本字典，供条件扩散使用。
