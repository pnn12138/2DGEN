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
- 当写入了 canonical 字段时，缓存元数据会记录 `coord_frame="canon"`，用于训练侧启用 geometry heads。

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
- 训练新增角度/Gram-condition 约束（`loss_angle`/`loss_cond`），并在日志中记录 angle_out_rate/cond_mean。
- 训练默认启用 vacuum loss（`vacuum_loss_weight=0.1`），采样/评估入口同步传入 `vacuum_min` 以强化 2D 真空约束。
- 训练日志新增 `chol_log_clamp_rate` 与 `z_norm` 统计，便于诊断晶格触边与几何尺度对齐。
- 厚度 `t` 的归一化在使用 split 时改为基于训练子集统计，避免数据泄漏。
- `z_norm` 默认噪声尺度下调以缓解 `loss_zn` 量级失衡（可通过 CLI 覆盖）。
- 当 `coord_frame` 元数据不一致时自动禁用 geometry heads，避免混用 raw/canon。
- 训练默认尝试加载 `c2db_quality.jsonl` 进行过滤，缺失时降级为不过滤。

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
- 可选采样修正：`--expand-vacuum`、`--expand-on-collision`、`--lattice-jitter`。

### 4.3 采样输出
- `samples.npz`（含 `z/frac/lattice/atom_mask`，以及条件记录）。
- `samples.npz` 记录 `chol_log_clamp_rate`（若使用 cholesky6 且设置了 log bounds），用于监控晶格触边比例。
- 可选导出 CIF（`--save-cif`），默认写入评估目录。

---

## 5) 评估流程
### 5.1 Tier‑0/1 评估
- `evaluate/eval_samples.py`：合法性（min_dist/体积/重复）、2D 统计（thickness/vacuum）。
- 条件匹配：exact match、L1 计数误差、组成相似度。
- 输出：`per_sample.jsonl` + `tier0_metrics.json` + `tier1_2d_metrics.json`。
- `vacuum_min` 默认 15.0，用于输出 `vacuum_ok_rate`；同元素最小距离缺失时写 `null` 以避免 NaN。

### 5.2 可视化
- `evaluate/plot_eval.py`：分布直方图与散点图。
- `evaluate/plot_compare.py`：与真实数据分布对比（精确 MIC）。

### 5.3 CIF 评估闭环（Phase 3）
- Tier-0 CIF 入口：`evaluate/eval_tier0_cif.py`。
- 条件校验：`evaluate/check_conditions.py`（formula/elements/spacegroup）。
- MatterSim 能量：`evaluate/mattersim_energy.py`（可选 relax）。
- 形成能计算：`evaluate/formation_energy.py`（参考能表）。
- 合并报告：`evaluate/merge_reports.py`。
- 统一入口：`evaluate/run_pipeline.py`（支持分步执行）。

---

## 6) Phase 0-3 进度对齐（基于当前实现）
### Phase 0：现状基线与评估规范化
- T0.1 固化 baseline 配置与指标格式：已落地 `twodgen/baselines/eval_run_001.md` 与 `twodgen/evaluate/eval_run_001.py`；评估产物固定为 `tier0_metrics.json`/`tier1_2d_metrics.json`/`per_sample.jsonl`，`valid_rate` 统一为 `tier0_metrics.json:valid_rate_eval`（不同于采样阶段 quick check）。
- T0.2 训练/验证集划分：已落地 `twodgen/data/create_c2db_split.py`（按 `n_atoms/t_bin/top_elem` 分层，输出分布差异检查），评估脚本通过 `cond_split=train|heldout` 区分统计。

### Phase 1：数据集治理与预处理
- T1.1 硬过滤规则：已落地 `twodgen/data/clean_c2db_2d.py`（原子数上限/真空层/跨真空成键），输出 `c2db_clean_report.json` 统计报告。
- T1.2 质量标签分桶：已落地 `clean_c2db_2d.py` 的 `quality_tags`/`quality_bucket` 与 `c2db_quality.jsonl`。
- T1.3 exp/theo 分桶：已落地 `clean_c2db_2d.py` 的 `source_bucket`（exp/theo/unknown）。

### Phase 2：生成有效率与物理合理性提升
- T2.1 几何可行性强化：已落地 `twodgen/scrip/train_tokens.py` 的 `min_dist` 训练惩罚 + collision curriculum（`--curriculum-collision`），采样端记录 pre/post collision 统计。
- T2.2 条件可控性修复：已落地 composition encoder + 成分一致性损失（`--comp-loss-weight`，`--comp-loss-mode=l1|cosine`），`counts_vector` 作为强条件注入。
- T2.3 二维物理合理性：已落地 cross-vacuum 检测与评估（token cache + `eval_samples`），采样支持 `--reject-cross-vacuum` 过滤；训练侧显式惩罚仍待补齐。

### Phase 3：评估体系完善
- T3.1 分层评估指标固化：已落地 `twodgen/evaluate/tier_definitions.md` + Tier-0/1 脚本；Tier-2 预留 `property_predict.py`（mock 或真实模型）。
- T3.2 消融实验设计：已提供 `twodgen/evaluate/ablation_matrix.json` + `run_ablation.py`，尚需实际运行与结果表整理。

---

## 7) 当前瓶颈与待解决问题（基于 200 epoch 指标）
- 2D 真空约束未真正生效：训练默认 `--vacuum-loss-weight=0.0`，导致 `valid_2d_rate` 极低、`cross_vacuum_rate` 偏高，需要在训练/采样/评估链路显式启用并对齐 `vacuum_min`。
- 晶格体积/角度分布塌缩：采样体积近似常数、角度集中在 90 度附近，需排查 lattice/Gram 是否被 `cell_init` 或条件常量覆盖，确认模型输出是否真正参与采样。
- 结构碰撞率高：Tier‑0 `valid_rate_eval` 仅 ~0.1，`min_dist` 低于阈值；需加强训练惩罚与采样 repulsion 并对齐 `eval_min_dist`。
- 训练日志存在 `min_dist_mean=Infinity`：需在训练统计中识别空邻接/异常 batch，记录样本索引以定位根因。
- 评估输出存在 NaN 字段与命名不一致：`min_dist_same_elem` 在无同元素样本时为 NaN，`per_sample.jsonl` 与实际产物命名需统一，避免下游统计失真。
