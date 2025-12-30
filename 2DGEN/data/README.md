# 2DGEN data utilities

本目录包含两条数据路径：在线填充的 `C2DBDataset` 与离线网格 `C2DBGridNPZDataset`（由 `prepare_c2db_grid.py` 生成）。

- 网格形状 `(3, max_atoms, 3*2*F)`，通道定义：
  1. `grid[0]`：原子序数除以 `atomic_scale`，沿宽度重复
  2. `grid[1]`：分数坐标的 torus sin/cos 编码
  3. `grid[2]`：晶格参数 `[a,b,c,alpha,beta,gamma]`，a/b/c 除以 `lattice_scale`（默认 10.0），角度（度）除以 `angle_scale`（默认 180.0），写入前 6 个宽度位置并广播到所有原子行

## C2DBDataset（在线预处理）
文件：`2DGEN/data/c2db_dataset.py`

- 逐行读取 `data/C2DB/c2db_summary.csv`，解析 CIF，超出 `max_atoms` 可选丢弃。
- 生成 `atomic_numbers`、`frac_coords`、`lattice_matrix`，并用 `_pad_1d`/`_pad_2d` 填充到固定长度；`atom_mask` 标记真实原子。
- 样本结构：`{"atomic_numbers", "frac_coords", "atom_mask", "lattice_matrix"}`（torch 张量），元数据存入 `C2DBMetadata`。
- `collate_fn` 简单堆叠；`get_metadata` 按索引取元信息。

## Token 预处理（A++ v3 可选）
文件：`2DGEN/data/prepare_c2db_tokens.py`

- 默认保留原有字段：`z`、`f`（分数坐标）、`atom_mask`、`lattice`、`gram6`。
- 开启 `--preprocess-v3` 后会额外缓存 canonical slab 表示（全部已 padding 到 `max_atoms`）：
  - `z_canon`：canonical 排序后的元素序列
  - `uvz`：`(u, v, z_norm)` 三元坐标
  - `u`, `v`, `z_norm`：单独存储便于检索
  - `t`、`a_hat`、`b_hat`、`n`、`lattice_param`、`counts_vector`、`order_idx`
- 同时写入 `preprocess_version=A++_v3` 与 `eps_area/eps_inv/round_prec/z_norm_clip` 供复现。

## 网格化预处理：prepare_c2db_grid.py
文件：`2DGEN/data/prepare_c2db_grid.py`

- `row_to_grid(cif_str, max_atoms, atomic_scale, torus_freqs, lattice_scale, angle_scale)`：
  - 若原子数超出 `max_atoms` 返回 `None`。
  - `grid[0]`：原子序数/`atomic_scale`
  - `grid[1]`：torus 编码分数坐标（默认频率 `1,2,4,8`）
  - `grid[2]`：晶格参数 `[a,b,c,alpha,beta,gamma]` 缩放后写入前 6 列并广播
- `build_dataset(csv_path, max_atoms, atomic_scale, limit, torus_freqs, lattice_scale, angle_scale)` 读取 CSV，逐行转换、堆叠。
- `main` 将结果保存为 npz：`x`、`material_id`、`torus_freqs`、`lattice_scale`、`angle_scale`。

生成与验证示例：
```bash
# 预处理
uv run python 2DGEN/data/prepare_c2db_grid.py \
  --csv data/C2DB/c2db_summary.csv \
  --out data/C2DB/ache/c2db_grid.npz \
  --max-atoms 24 --torus-freqs 1 2 4 8 --atomic-scale 100 --lattice-scale 10 --angle-scale 180

# 简单检查
uv run python - <<'PY'
import numpy as np
npz = np.load('data/C2DB/ache/c2db_grid.npz')
print('x shape:', npz['x'].shape, 'torus_freqs', npz['torus_freqs'], 'lattice_scale', float(npz['lattice_scale']))
PY
```

## C2DBGridNPZDataset（加载离线网格）
文件：`2DGEN/data/c2db_dataset.py`

- 读取 `npz` 中的 `x`，可选返回 `material_id`；若存在则保留 `torus_freqs`、`lattice_scale`、`angle_scale` 属性供训练/导出参考。
- `__getitem__`：存在 `material_id` 时返回 `(sample, material_id)`，否则仅返回 `sample`。
- 可直接与 `2DGEN/scrip/train.py`/`2DGEN/scrip/sample_and_export.py` 搭配。***
