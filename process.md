# 2dgen 项目细节概览

## 一、数据集与预处理（C2DB）
- 原始数据：`data/C2DB/c2db_summary.csv`，每行包含 CIF 文本与材料元数据。
- 在线数据集：`twodgen/data/c2db_dataset.py`
  - `C2DBDataset`：逐行解析 CIF，生成 `atomic_numbers`、`frac_coords`、`lattice_matrix` 并按 `max_atoms` padding，`atom_mask` 标记真实原子。
  - `C2DBAtomDataset`：在 `C2DBDataset` 基础上额外输出 Gram6（`g`）用于 token 扩散（`g_scale` 缩放）。
- Token 缓存预处理：`twodgen/data/prepare_c2db_tokens.py`
  - 输入 CIF，输出 `.npz`（默认 `data/C2DB/cache/c2db_tokens_2d_based.npz`）。
  - 统一晶胞约定：`cart = frac @ lattice`，`gram6 = lattice @ lattice^T`（写入 `gram6_convention=row_lattice`）。
  - A++ v3 预处理（默认开启）：`twodgen/data/preprocess.py`
    - 2D slab 规范化：依据原子 PCA/法向确定 `n`，对 `(a,b)` 基做 2D unimodular 规约，得到 canonical `a_hat/b_hat` 和 `lattice_canon`。
    - 将原子投影到 2D 面内，计算 `(u,v)`，用圆周均值对齐相位；沿法向得到 `z_norm` 与厚度 `t`。
    - 生成 `uv_angle=(cos 2πu, sin 2πu, cos 2πv, sin 2πv)`、`lattice_param`（log area + 2D Cholesky 参数）、`counts_vector`、`order_idx` 等条件字段。
  - 输出字段：`z/f/atom_mask/lattice/gram6` +（可选）`z_canon/f_canon/uvz/uv_angle/u/v/z_norm/t/a_hat/b_hat/n/lattice_param/counts_vector/order_idx/lattice_canon/gram6_canon`。

## 二、模型与训练（Token 扩散默认路线）
- 核心模型：`twodgen/model/atom_transformer.py` + `twodgen/model/atom_denoiser.py`
  - AtomTransformer：以 `(Z, F, g)` token 为输入。
    - 元素 `Z` 用 embedding；`F` 用 Fourier + MLP 编码。
    - 通过 kNN 邻居图 + GatherAttention 做原子-晶胞联合注意力；支持 2D PBC（默认 `pbc_mask=(1,1,0)`）。
    - 条件输入：`cond`（如 `lattice_param/t`）和 `counts_vector`（成分条件）；可选 composition encoder。
  - AtomDenoiser：包装 diffusion/flow 损失与采样逻辑。
    - `AtomVelocityLoss`（`twodgen/common/atom_diffusion.py`）在训练时用 logit-normal 采样 `t`，预测 `x0` 并换算为速度 `v`。
    - 支持几何附加变量（`uv_angle`, `z_norm`, `lattice_param`, `slab_t`），并支持不确定性权重（learnable `s_*`）。
    - 可选 `min_dist` 训练惩罚（避免过近原子）。
- 训练入口：`twodgen/scrip/train_tokens.py`（默认参数在脚本内固定）。

## 三、采样实现
- 入口脚本：`twodgen/scrip/sample_tokens.py`
  - 采样方法：Euler / Heun（默认 Heun），`num_sampling_steps` 与 `neighbor_update_steps` 可调。
  - 晶胞与几何约束：
    - 支持 `cell_init`、`reduce_lattice`、`niggli_reduce`；
    - 支持 `min_dist_project`（采样过程中最小距离投影）；
    - 可对 `z` 进行 temperature/top-k/top-p 采样。
  - 输出：`samples.npz` + 可选 CIF，并默认触发 `eval_samples` 写出评估结果到 `out-dir/eval/`。

## 四、评估逻辑
- 主评估脚本：`twodgen/evaluate/eval_samples.py`
  - 输入 `samples.npz`，输出 `per_sample.jsonl`、`tier0_metrics.json`、`tier1_2d_metrics.json`。
  - Tier-0（基础有效性）：
    - 失败原因：空样本、`n_atoms < 3`、体积越界、Gram 非 SPD、最小距离低于阈值（`min_dist_cut`）、重复坐标比例过高、角度越界（alpha/beta/gamma 超出 30–150 度）。
    - 统计：`min_dist/volume/cond/n_atoms/angles`，元素计数，条件匹配（`counts_vector`）等。
  - Tier-1（2D 结构）：
    - 按最大晶格向量作为 `c`，计算 `thickness/vacuum`，并检测“跨真空成键”（在 `c` 非周期方向出现跨胞近邻）。
    - 统计 `valid_2d_rate`、`cross_vacuum_rate`、`gcc_ratio`（最大连通分量占比）、`anisotropy`（`c_len / mean(a,b)`）。

## 五、当前模型性能（`outputs/eval_run_001/eval`）
- 样本数：2000
- Tier-0 指标：
  - `valid_rate_eval`: 0.5365（失败主因：`collision=914`、`low_atoms=16`）
  - `min_dist`：mean 1.7285、p10 0.8314、p90 2.6648
  - `volume`：mean 703.8797、p10 316.3361、p90 1245.6885
  - `n_atoms`：mean 8.625、p10 3、p90 16
  - `angle_alpha/beta/gamma` 平均约 91.15 / 90.83 / 83.30
  - 条件匹配（counts_vector）：exact 0.0，L1 误差均值 15.322，comp_cosine 均值 0.1487
- Tier-1（2D）指标：
  - `valid_2d_rate`: 0.465
  - `thickness`：mean 13.6911（p10 7.1812，p90 20.5714）
  - `vacuum`：mean 7.8457（p10 4.5270，p90 11.9504）
  - `cross_vacuum_rate`: 0.1415
  - `gcc_ratio`: mean 0.4684；`anisotropy`: mean 3.8708

（如需更细粒度统计可查看 `outputs/eval_run_001/eval/per_sample.jsonl`。）
