# twodgen 项目脉络（当前实现）

> 目标：记录当前 token 路线的真实实现脉络，便于训练/采样/评估对齐。

---

## 0) 总览
- 主路线：`Z/F/g` token 扩散（`g` 为 `gram6` 或 `cholesky6`，内部按 `g_scale` 缩放）。
- 预处理：A++ v3 canonical 生成（`data/preprocess.py` + `data/prepare_c2db_tokens.py`）。
- 条件：默认 `counts_vector,lattice_param,t`（脚本默认值可修改）。
- 几何头：`uv_angle/z_norm/lattice_param/t` 可选，按数据字段启用。
- PBC：默认 slab 2D（`pbc_mask=1,1,0`），邻居图在线构建。

---

## 1) 数据与预处理
### 1.1 预处理流程（A++ v3）
- 从 CIF 解析 `cell + pos_cart + Z`。
- 计算 slab 法向、面内投影、2D reduction 得 `a_hat/b_hat`。
- torus mean shift 去面内平移 gauge。
- 厚度 `t` 与 `z_norm` 归一化并 clip。
- `lattice_param`：log-area + shape（2D log-Cholesky）。
- canonical 排序（`Z, z_norm, u, v`）。

### 1.2 预处理缓存字段（npz）
- 主输入：`z, f, gram6, atom_mask`。
- canonical 附加：`f_canon, lattice_canon, gram6_canon, order_idx/order_inv`。
- 几何字段：`uv_angle, z_norm, t, lattice_param, counts_vector` 等。
- 约定：`gram6_convention="row_lattice"`，`cart = frac @ lattice`。

### 1.3 Dataset 读取
- `C2DBTokenNPZDataset`：读取 npz，按 `coord_frame` 选择 raw/canon。
- 若请求 `coord_frame=canon` 但缺失 `f_canon/gram6_canon`，会回退为 `coord_frame_actual=raw`。
- `align_atoms=True` 时依据 `order_idx` 对齐 per-atom 字段。
- `counts_vector` 默认长度 118（`Z-1` 映射）。

---

## 2) 训练流程（token 扩散）
### 2.1 训练入口
- `scrip/train_tokens.py`：构建数据集 → 模型 → 训练循环。
- 保存 `model_state_dict`/`ema_state_dict` + `model_config` + `diffusion_config` + `cond_config`。

### 2.2 条件向量
- `cond_fields` 默认 `counts_vector,lattice_param,t`。
- `counts_vector` 按 `max_atoms` 归一化；连续字段可用 `cond_stats` 做 z-score。
- `cond_drop_prob` 训练时生效（同时 drop `cond_vec` 与 `counts_vector`）。

### 2.3 扩散目标
实现位置：`common/atom_diffusion.py::AtomVelocityLoss`
- 连续变量预测 `x0`，训练中换算为 `v` 监督。
- `Z` 走 mask diffusion（交叉熵）。
- 可选 flow-matching（`mode=flow`）。

### 2.4 训练侧几何与碰撞约束
- geometry heads 可选（`uv_angle/z_norm/lattice_param/t`）。
- 训练附加 `min_dist` penalty（`min_dist_train_weight`）。
- 训练日志包含 `min_dist` 分布与 collision rate。

---

## 3) 模型结构（AtomTransformer + AtomDenoiser）
### 3.1 AtomTransformer（主干）
- token：CELL + ATOM；ATOM 由 `Z` embedding + torus 坐标编码。
- 邻居：在线 MIC kNN（2D PBC），支持 dual-graph 与 wrap embedding。
- 条件注入：DiT 风格 AdaLN + gating（block 内部）。

### 3.2 composition encoder（已落地）
- `counts_vector` → element embedding pooling → `cond_comp`。
- `cond = w_time*cond_time + w_vec*cond_vec + w_comp*cond_comp`，`w_comp` 初始为 0。

---

## 4) 采样流程
### 4.1 采样入口
- `scrip/sample_tokens.py`：加载 checkpoint +（可选）`cond-npz`。
- 默认优先加载 EMA；打印 `coord_frame`/`g_scale`/`pbc_mask` 兼容性信息。

### 4.2 采样迭代
- 支持 `euler/heun`；`Z` 可 `temperature/topk/topp` 采样。
- `project_geometry` 只投影 `uv_angle/z_norm`；`project_each_step` 可额外投影 `frac/gram6`。
- 采样末尾做 min_dist repulsion，输出 pre/post 统计。

### 4.3 采样输出
- `samples.npz`（含 `z/frac/lattice/atom_mask`，以及条件记录）。
- 可选导出 CIF（`--save-cif`），默认写入评估目录。

---

## 5) 评估流程
### 5.1 Tier‑0/1 评估
- `evaluate/eval_samples.py`：合法性（min_dist/体积/重复）、2D 统计（thickness/vacuum）。
- 条件匹配：exact match、L1 计数误差、组成相似度。
- 输出：`per_sample.jsonl` + `tier0_metrics.json` + `tier1_2d_metrics.json`。

### 5.2 可视化
- `evaluate/plot_eval.py`：分布直方图与散点图。
- `evaluate/plot_compare.py`：与真实数据分布对比（精确 MIC）。
