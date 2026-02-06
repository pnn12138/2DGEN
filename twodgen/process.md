# twodgen 项目脉络（当前实现）

> 目标：把 token 轨道的端到端实现写清楚——数据如何打标整理、模型/训练/采样/评估如何串联，每一步都指向具体模块/函数，方便后续调试与验证。

---

## 0) 架构概览
- 主轴是 `Z/F/g` token diffusion：`Z`（原子序号）、`F`（fractional coords）、`g`（Gram6 或 cholesky6）分成一个序列，交给 `AtomTransformer` 进行跨原子 self-attn，丁点细节由 `AtomDenoiser` 封装。
- 所有几何变量以 canonical frame（`coord_frame="canon"`）和 align_atoms 保障可比较的 per-atom 顺序；`prepare_c2db_tokens.py` 选项 `--align-atoms` / `--coord-frame canon` 控制是否写入 `order_idx/order_inv`。
- 条件默认是 `counts_vector,lattice_param,t`（训练脚本`cond_fields`/`cond_normalize_fields`），并走 composition encoder 为元素计数提供额外 embedding。
- slab 2D PBC（`pbc_mask=(1,1,0)`）贯穿数据、训练、采样与评估，任何需要 3 层 2D 物理的逻辑都会显式参考 `twodgen/common/crystal.py` 提供的 `frac_mic_dist`、`gram6<->cholesky`、`build_knn` 等工具。

### 0.1 权威 lattice 表示规范（2026-02-01）
> 统一“修复入口”和投影/损失口径：**权威表示为 `gram6`（scaled Gram）**。模型内部可选 `cell_rep=cholesky6` 作为参数化，但所有投影/角度/cond 计算都先转换为 `gram6` 再处理。

| 位置 | 入口 → 输出 | 说明 |
| --- | --- | --- |
| `twodgen/model/atom_denoiser.py::_cell_to_gram6` | `cell` → `gram6` | cell_rep=cholesky6 时转成 Gram6，作为权威表示。 |
| `twodgen/model/atom_denoiser.py::_gram6_to_cell` | `gram6` → `cell` | 采样/投影后回写到 cell_rep。 |
| `twodgen/common/crystal.py::project_gram_cond_spd` | `gram6` → `gram6` | SPD + cond 投影，只作用在权威表示。 |
| `twodgen/common/crystal.py::gram6_to_lattice` | `gram6` → `lattice` | 角度/cond 统计、输出 CIF 时的最终物理晶格。 |
| `twodgen/common/crystal.py::lattice_to_gram6` | `lattice` → `gram6` | 评估或回写时统一回到权威表示。 |

---

## 1) 数据与预处理

### 1.1 清洗与质量标注（`twodgen/data/clean_c2db_2d.py`）
- `run_cleaning` 迭代 CSV，每条 CIF 经过 `analyze_cif`，先计算 slab vacuum axis/length（`_choose_vacuum_axis`），再用 `_mic_dist_and_shifts` 跨真空/2D MIC 判断 `min_dist`、cross-vacuum bond、thickness/vacuum。
- 根据 `C2DB2DQualityConfig` 里的 `max_atoms`/`min_vacuum`/`collision_risk_cut` 产生 `hard_fail_reasons` 与 `quality_tags`，最终把每条带 `hard_pass`、`quality_bucket`（good/risk/bad）写入 `c2db_quality.jsonl`，并支持 `source_bucket`、`quality_tags`、`source_type` 等字段的统计（`c2db_clean_report.json` 与 `clean_c2db_2d.py --report` 输出）。
- 该脚本还能 split 生成 `quality_bucket`、`source_bucket`、`min_dist` 等指标的 histogram 供 landmarks 参考。

### 1.2 预处理 + canonical 化（`twodgen/data/preprocess.py` + `prepare_c2db_tokens.py`）
- `preprocess_cartesian` 接受 `cell/pos_cart/z_numbers`，用 PCA 或 `cell` 面内向量求 slab normal，用 `_reduce_2d_basis` 找出 2D 基底，扔进 `_least_squares_uv` 得 `u/v`（torus 上做 mean shift 去 gauge）。
- 归一化后算 `z_norm`（厚度 `t` = upper quantile – lower quantile），再稳定化 `lattice_param`（log-area + 2D Cholesky with clamp），输出 canonical atomic order、`uv_angle`、`uvz`、`z_norm`、`t`、`counts_vector`、`order_idx/order_inv`。
- `prepare_c2db_tokens` 每行 CIF 走 `row_to_tokens`：先 pad（`data.utils.pad_1d/pad_2d/pad_2d4`）/mask；再用 `_mic_dist_and_shifts` 计算 `min_dist`、`cross_vacuum_bond`、`thickness`/`vacuum`；并调用 `preprocess_cartesian` 生成 canonical `z`, `f`, `gram6`, `lattice`，把 canonical per-atom fields 写进 npz（`f_canon`, `atom_mask_canon`, `gram6_canon`, `uv_angle`, `uvz`, `z_norm`, `lattice_param`）。
- 最终 `np.savez` 包含 `z,f,atom_mask,gram6,lattice`、canonical 版本、`counts_vector`, `slab_thickness/vacuum`, `min_dist/collision_risk`、`schema_version`, `coord_frame`, `g_scale`, `max_atoms` 等元数据，`npz` 里的 `gram6_convention="row_lattice"` 保证 `g @ g^T` 变换兼容。

### 1.3 数据集类与切分
- `twodgen/data/c2db_dataset.py` 提供 `C2DBTokenNPZDataset` 直接读 npz，按 `coord_frame` 选择 `raw`/`canon`，`align_atoms` 受 `order_idx` 约束。
- `C2DBTokenNPZDataset` 也会把 `extra` 字段（只要 shape[0] = n_rows）缓存进 `dataset.extra`，供 `train_tokens._compute_dataset_min_dist` 直接重用 `min_dist`，避免重算 `frac_mic_dist`。
- `IndexedDataset` wrap 任何 base dataset，加一个 `index` 字段供训练/日志使用； `IndexedDataset.collate_fn` 特殊处理 `index`，`_unwrap_indexed_dataset` 在 `_compute_dataset_min_dist` 中用来跳过该字段，防止 `torch.stack` 报错。
- 划分用 `twodgen/data/create_c2db_split.py` 按 `n_atoms`, `element counts`、`vacuum` 等做 stratified split，生成 `trainsplit/heldout` JSON；训练脚本可用 `--split-json/--split` 指定 `train`/`heldout` 子集，再通过 `_filter_indices_by_quality` 结合 `c2db_quality.jsonl` 的 `quality_bucket` 做 `hard_pass`/`risk` 过滤（`quality_map` 由 `_load_quality_map` 读取）。

### 1.4 统计与可视化工具
- `twodgen/data/splits.py` 里封装 `select_split_indices/load_c2db_split/validate_split_indices`，`twodgen/data/torus.py::torus_encode` 提供 `uv` embedding。
- `twodgen/data/clean_c2db_2d.py` 内的 `_thickness_vacuum`、`_mic_dist_and_shifts` 也会在训练/采样中复用（通过 `twodgen/data/prepare_c2db_tokens.py` 的导入），确保 metrics 一致。

---

## 2) 训练管线

### 2.1 入口与配置（`twodgen/scrip/train_tokens.py`）
- CLI 解析大量参数：`--npz`/`--csv` 选择数据源，`--align-atoms`/`--coord-frame` 控制 canonical frame，`--min-dist-train-cut/--min-dist-train-weight` 配置碰撞 penalty，`--curriculum-collision`/`--filter-min-dist-below` 控制样本在线筛选，`--vacuum-loss-weight/--vacuum-min` 绑定 vacuum 惩罚。
- 数据集构建：训练仅支持 `C2DBTokenNPZDataset`（`--npz`），并 wrap 成 `IndexedDataset` 方便日志追踪；base dataset 的 `g_scale`、`coord_frame_actual` 等 metadata 直接注入日志/警告。
- 训练前预处理：质量过滤（`quality_map`）、curriculum indices（`_curriculum_indices`）根据 `min_dist_all`（`_compute_dataset_min_dist` 支持直接用 cached + `frac_mic_dist`）决定 sample order；`_normalize_cond_stats`、`collect_run_metadata` 记录 `cond_stats`、`run` ID，训练开始便写入 `outputs/checkpoints/<run>/config.json`。
- `train_one_epoch` 每 step：从 loader 得 `z`, `frac`, `atom_mask`, `gram6`，同时 `batch_indices`（index）用于在 `min_dist_inf` 记录异常样本；每 `log_interval` 计算 `chol_log_clamp_rate`（通过 `gram6_to_cholesky6` 判断是否 hit log bounds）、`min_dist_mean/p10/collision`、`loss_uv/loss_zn/loss_lat` 等 metrics，追加写入 `train_metrics.jsonl`，还有 `train_metrics/config.jsonl`＋`run_metadata`。
- 参数优化器使用 `AdamW` + cosine LR schedule（`_compute_lr/_adjust_lr`），EMA 选项可通过 `--ema` 开启；`save_checkpoint` 负责写 `atomdenoiser_last/best.pt`，`model_config`, `diffusion_config`, `cond_config` 三份配置一同持久化。

### 2.2 losses & constraints
- 基础 diffusion loss：`AtomDenoiser._predict_velocity` 目标 `x0`，通过 `AtomVelocityLoss`（`common/atom_diffusion.py`）把 `x0` 转成 velocity、Gram residual、`Z` logit；`z` 走 mask diffusion（`logits_z`）并结合 `ignore_tokens`.
- Geometry heads：`uv_angle`, `z_norm`, `lattice_param`, `t`（thickness）在 `AtomDenoiser._project_geometry_step` 被 `atom_mask` 约束；`project_each_step` 控制是否在每个采样 step 投影 `frac/gram6`，训练中 `use_geometry_fields` 依赖 dataset 是否有 canonical fields。
- Collisions/vacuum：训练有 `min_dist` penalty（`min_dist_train_weight`），对所有 `dist < min_dist_cut` 的原子对累计惩罚（上三角避免重复）；`vacuum_loss_weight` 默认走 `hinge(c_axis)`（`vacuum_loss_mode="c_axis"`，`max(0, c_min - c_axis)`），`loss_vacuum` 的 vacuum axis 与 z-clamp 共用 `choose_vacuum_axis`（默认“最长轴”），日志里输出 `loss_vacuum`/`vacuum_gap_mean`/`c_len_mean` 便于诊断真空尺度。
- Cross-vacuum 训练惩罚：`loss_cross_vacuum` 使用 3D MIC + vacuum axis shift 近似检测跨真空成键风险（`cross_vacuum_bond_cut`），并与 `atom_mask/cond_mask` 正确屏蔽；日志输出 `loss_cross_vacuum` 与 `cross_vacuum_rate`。
- `chol_log_relax`/`chol_log_min/max` 在 `AtomDenoiser._project_step` 中控制 cholesky clamp，`AtomTransformer` 会在 `cholesky6_to_lattice` 与 `lattice_to_gram6` 之间加 clamp；`chol_log_clamp_rate` 统计 clamp 命中率，帮助诊断 lattice collapse。
- 条件 loss：条件向量 `cond` 由 `counts_vector`/`lattice_param`/`t`（训练可用 `cond_drop_prob` 做 dropout），再加 composition encoder（`AtomTransformerConfig.use_comp_encoder`）产生 `cond_comp`；`loss_comp` 由 `comp_loss_weight`（l1/cos）控制；采样支持 CFG（`--cfg-scale>1`）增强成分对齐。
- 晶格条件数惩罚：`lambda_cond` 对应的是晶格 Gram 条件数（condition number）超出 `cond_max` 的 hinge 惩罚，指标名为 `loss_cond_number`（用于避免与“条件向量 cond”的语义混淆），并配套记录 `pred_cond_mean`。
- 物理约束调度：增加了 `twodgen/loss/schedule.py::LossWeightScheduler`，默认 warm-up 15k 步、采用 `sigmoid` 曲线、把 `vacuum/cond/chol_bound/expand_collision/volume/c_len` 等物理项纳入统一调度；`train_tokens.py` CLI 默认 `--volume-loss-weight=0.1`、`--c-len-loss-weight=0.05` 并提供 `--loss-weight-schedule`（linear/sigmoid/cosine），训练配置会持续向 `train_metrics.jsonl` 输出 `loss_volume/loss_c_len`、`loss_chol_bound`、`vacuum_gap_mean`/`c_len_mean`、`lambda_*` 等指标，便于验证每项 constraint 是否已激活。
- 全局 lattice 约束：在 `twodgen/common/atom_diffusion.py` 里新增 volume/c_len penalty，结合已有的 `loss_chol_bound` 与 expand-on-collision，通过 `lambda_volume`/`lambda_c_len`、`c_len_min/vacuum_min` 等参数强制生成的晶胞满足面积/体积和 c 轴真空标准；expand-on-collision 惩罚现在根据距离严重度放大，并且 `LossWeightScheduler` 会在训练早期就让这些约束逐步释放，不再等到后期才“补救” collapse。

### 2.3 数据增强与后处理
- `min_dist` repulsion在训练/采样都可启用：`AtomDenoiser._apply_min_dist_repulsion` 迭代 `min_dist_iter` 次，`strength` 控制 push 散度，`frac` 需要 center-of-mass 修正避免漂移。
- 训练曲线写盘：`train_metrics.jsonl` 以 JSON lines 按 `log_interval` 写入 `loss/min_dist/chol_log_clamp/geometry losses`，还有 `per-step` stats（`min_dist_inf`, `min_dist_low_atoms` + sample indices）。训练侧可选接入 TensorBoard/W&B，记录 `min_dist/vacuum_gap/chol_diag` 分布、主要 loss 与梯度范数，并在前 1 万步触发异常告警。
- 所有 cond/geometry stats 通过 `_normalize_cond_stats` 记录 `cond_stats`/`cond_fields`/`t_stats`，便于 later normalization（例如 `t` 只用训练 split 的 mean/std，避免泄漏）。

---

## 3) 模型结构详解

### 3.1 AtomTransformer 主干
- `AtomTransformer` 接受 concat [cell token + atoms]，Cell token 与 atoms 共享 embedding；`GatherAttention` 分别计算 cell→atoms、atoms→neighbors 的 attention，`bias_nbr`/`bias_atom_cell`/`bias_cell_atom` 由 `build_knn` 提供 neighbor bias/atom-cell bias/graph bias。
- 位置编码：`torus_encode` 和 `rbf_expand` 为 fractional coords/Gram distances 编码；`modulate` 结合 `cond` 生成 AdaLN 风格的 shift/scale/gate，`AtomBlock` 包含 attention + MLP + gating 组成。
- 结构上支持 `dual_graph`、`wrap_embed_dim`、`edge_type_gating`、`cache_neighbors`、`edge_type_dim`、`element_ids` 等开关，allow customizing kNN or gating for ablations。

### 3.2 AtomDenoiser 包装
- 在 `twodgen/model/atom_denoiser.py` 中，`AtomDenoiser` 先把 `frac`/`gram6`/`z` embed 进 `AtomTransformer`，再由 `AtomTransformer` 的 output 拆成 `vel`、`uv_angle`、`z_norm`、`lattice_param`、`t`（`geom_preds`）。
- `project_geometry_step` 使 `uv_angle`/`z_norm` 归一；`_project_step` 会对 `gram6` 做 cholesky clamp，投影回 `frac`/`cell`，不再额外做体积/条件数的 clip。
- `loss_min_dist`、`loss_angle`、`loss_cond_number` 通过 `metrics` dict 和 training loop 进行 logging；`pred_cond_mean`/`pred_angle_out_rate` 记录 geometry 侧状态。`_predict_velocity` 也可能返回 `_apply_min_dist_repulsion` 之后的 frac。
- `AtomDenoiser` 支持 `min_dist_iter`/`strength`/`p_mask_min|max`/`lambda_uv|lambda_zn|lambda_lat` 等 diffusion hyper-params，在 `denoiser_cfg.diffusion` 中配置。

---

## 4) 采样链路

### 4.1 `twodgen/scrip/sample_tokens.py`
- 加载 checkpoint（优先 EMA），并打印 `coord_frame`, `g_scale`, `pbc_mask` 兼容性报警；`--samples` 决定生成数量，`--cell-init`/`--cell-init-scale` 控制初始 lattice。
- 噪声 schedule 支持 `euler`/`heun`，`AtomDenoiser` 的 `_heun_step` 对 `frac`, `gram6`, `t` 进行迭代，完成后 `project_geometry`/`project_each_step` 再修正 canonical fields。
- `min_dist` repulsion、`--expand-vacuum`, `--expand-on-collision`, `--lattice-jitter` 等采样修正直接调用 `AtomDenoiser._apply_min_dist_repulsion`，并记录 repulsion 前后 `min_dist`/`collision` 统计。
- 最终输出 `samples.npz`（`z/frac/lattice/atom_mask` + `cond_counts_vector` + `chol_log_clamp_rate` + `batch_indices`），并可选 `--save-cif` 生成 CIF；每个 batch 也写 `per_sample`/`tier*_metrics` 供后续 eval。
- 采样结果始终包含 `energy_mlip/relaxed_flag/min_dist_relax` 字段：未启用 relax 时写入 NaN/0 作为占位，启用时填入真实松弛结果与 min_dist。

### 4.2 采样监控
- `samples.npz` 记录 `chol_log_clamp_rate`（`AtomTransformer` 中 clamp 触发统计），`min_dist_before/after` 由 `eval_samples` 使用 `frac_mic_dist` 统一计算。
- `outputs/samples_tokens` 结构里有 `samples.npz`, `eval`, `logs` 目录，`eval_samples` 还在 `per_sample.jsonl` 中写 `cond_matched` 等 200 条样本的详细字段。

---

## 5) 评估与指标

### 5.1 `evaluate/eval_samples.py`
- 对 `samples.npz` 逐行复现 `AtomDenoiser` 的 `frac`, `lattice`，用 `_min_dist_and_shifts` 计算 exact MIC min_dist，并按 `min_dist_cut=1.5`/`bond_cut`/`dup_eps` 做过滤。
- Tier 0：`valid_rate_eval`、`valid_criteria`、`min_dist/min_dist_collision/min_dist_same_elem`、`volume/cond/angle` 分布；Tier 1：`valid_2d_rate`, `thickness`, `vacuum`, `cross_vacuum_rate`, `vacuum_ok_rate`。
- `per_sample.jsonl` 记录 `valid`, `valid_2d`, `fail_reason`, `cond_exact_match`, `cond_l1`, `vacuum`, `cross_vacuum_bond`，没有重复元素时 `min_dist_same_elem=null`，避免 NaN；旧的 `per_sanmple.jsonl`/`tier*_metric.jsonl` 在写入时自动改名（手动 rename 已统一），确保历史目录与当前输出一致。
- `per_sample.jsonl` 同步写入 `energy_mlip/relaxed_flag/min_dist_relax`，`tier0_metrics.json` 额外汇总 `relaxed_rate` 与 `min_dist_relax` 分布，避免能量/松弛缺失导致指标不可追踪。
- 采样评估链路也会记录 `cond_counts_source`（来自 `twodgen/scrip/sample_tokens.py`）以证明 `cond_match` 不是用生成结果“回填”，`evaluate/eval_samples.py` 进一步检测如果 `cond_exact_match` 全 1 且没有真实目标则打上 `suspect_all_match` 标记，方便后续追踪是否有条件欺骗。

### 5.2 评估扩展
- `evaluate/plot_eval.py` + `plot_compare.py` 接 pipe 里的 `tier*_metrics.json` 画分布图，`evaluate/run_pipeline.py` 支持 tier-by-tier 运行（Tier0/1/2/3）。
- `property_predict.py` 目前是 placeholder，用于 Tier-2 未来引入 formation energy 评估；`evaluate/check_conditions.py` 与 `evaluate/mattersim_energy.py` 仍在整理中。

### 5.3 指标、日志与历史
- 训练/采样都写 `train_metrics.jsonl`/`per_sample.jsonl`，`train_metrics` 目录里还存在 `config.jsonl`, `train_metrics.jsonl` 记录 `min_dist`, `collision_rate`, `chol_log_clamp_rate`、`cond_mean` 等 line-by-line metrics；历史日志里我们已把旧 typo 改名，保证脚本 `/train_metrics/tier0_metrics.json` 找到正确文件。

---


## 6) 指标速览（2026-01-29）
- `outputs/eval_run_001/eval/tier0_metrics.json` 的 2000 样本（通过 `twodgen/evaluate/eval_samples.py`）显示 `valid_rate_eval`≈0.5365、`min_dist` mean 1.73 / p10 0.83 / p90 2.66、`volume` mean 703、`n_atoms` mean 8.6、`angle` 平均 ~91/90/83，以及 `counts_vector` exact-match 0.0、L1 误差平均 15.3、comp_cosine 0.149。
- 同一输出的 Tier-1 统计：`valid_2d_rate`≈0.465、`thickness` mean 13.7、`vacuum` mean 7.85、`cross_vacuum_rate`≈0.1415、`gcc_ratio` mean 0.468、`anisotropy` mean 3.87；详情可见 `outputs/eval_run_001/eval/per_sample.jsonl`。
- 评估“尺子”由 `twodgen/evaluate/eval_samples.py` 维护：在 `outputs/samples_tokens/baseline_20260129_raw` 生成的 100 个样本（`--no-project-each-step --no-min-dist-project`）上，valid_2d_rate≈0.70，记录 `formation_energy`/`cond_exact_match` 以便分层比较。
- 已集成 CHGNet relax（`twodgen/scrip/sample_tokens.py` 的 `--relax` + `relax_batch`），采样后强制真空扩展、写入 `energy_mlip/relaxed_flag/min_dist_relax` 并输出 `relaxed/*.cif`。20 样本对比（seed=0）显示 `relax_steps=20/100` 时 collision 0.35→0.30、valid_2d 0.65→0.70；z-clamp（`--relax-target-area-per-atom`/`--relax-flatten-z` 组合）后反而 collision=0.55、valid_2d=0.45，提示密度/压扁必须更温和。CHGNet 仍报告 isolated atoms，说明架构还需更强的 early-stage 约束。
- Phase2 统一验证（旧 checkpoint `outputs/checkpoints/20260122_134725/atomdenoiser_last.pt`，50 samples）：`cfg_scale=1.0` (`outputs/samples_tokens/20260122_134725_phase2_cfg1`) 的 `valid_rate_eval/valid_2d_rate`=0.02，`vacuum_ok_rate`=0.08；`cfg_scale=2.0` (`outputs/samples_tokens/20260122_134725_phase2_cfg2`) 的 `valid_rate_eval/valid_2d_rate`=0.08，`vacuum_ok_rate`=0.10。两次均 `cond_exact_match=1.0`（条件来自 target counts）。

## 7) 当前进度与下一步
### 7.1 已验证能力（根据代码）
- `twodgen/data/clean_c2db_2d.py` 负责 CSV 逐条分析、`C2DB2DQualityConfig` 计算 `min_dist`/`vacuum`/`cross_vacuum_bond`、`quality_bucket` 与 `quality_tags`，并把 `c2db_quality.jsonl`、`c2db_clean_report.json` 连同 histogram 写出，供训练/采样/评估共享。
- `twodgen/data/preprocess.py` + `prepare_c2db_tokens.py` 做 canonical 化：`preprocess_cartesian` 计算 slab 法向、`u/v` 坐标、`z_norm/t`、`lattice_param`，`row_to_tokens` 会产出含 `atom_mask_canon`/`gram6_canon`/`uv_angle` 等字段的 `.npz`，`coord_frame`/`order_idx` 保证 geometry head 可复用。
- `twodgen/data/c2db_dataset.py` 与 `twodgen/data/create_c2db_split.py` 搭配 `C2DBTokenNPZDataset`、`IndexedDataset`、`select_split_indices`，按 atom counts/vacuum/quality 做 stratified split，并在 `train_tokens` 中通过 `quality_map`/`curriculum`/`min_dist` 缓存复用 `frac_mic_dist` 统计。
- `twodgen/scrip/train_tokens.py` 解析 `IndexedDataset`、`--quality-jsonl`、curriculum、`vacuum_loss`、`comp_loss`、`min_dist` 策略，训练时通过 `collect_run_metadata` 写出 `outputs/checkpoints/<run>/config.json`，日志写入 `train_metrics.jsonl`（`chol_log_clamp_rate`、`min_dist_mean/p10`、`loss_uv/loss_lat`、`per-step min_dist_inf` 等）。`LossWeightScheduler`（`twodgen/loss/schedule.py`）对 `lambda_vacuum`/`lambda_cond`/`lambda_chol_bound`/`lambda_expand_collision` 做 sigmoid warm-up，实现更平滑的 constraint 激活。
- `twodgen/scrip/sample_tokens.py` 已实现 `_expand_and_center_vacuum`、`_rescale_inplane_for_density`、`_flatten_along_c` 等辅助函数，支持 `--cell-init`/`--reduce-lattice`/`--min-dist-project`/`--neigh-updates`，并在采样结束后调用 `relax_batch`（依赖 `chgnet` + `ase`）写回 `energy_mlip`、`relaxed_flag`、`min_dist_relax`，同时记录 `cond_counts_source` 便于评估验证 `cond_exact_match`。
- 评估链路以 `twodgen/evaluate/eval_samples.py`（exact MIC `min_dist_cut`、cross-vacuum/bond 检测、`per_sample.jsonl`、`tier0/tier1` 输出）、`run_pipeline.py`、`evaluate/check_conditions.py` 为核心，`per_sample` 中额外写 `suspect_all_match`/`cond_counts_source`，保证 `cond` match 不会被“回填”。
- 所有训练/采样/评估都会写 `train_metrics.jsonl`、`per_sample.jsonl`、`tier*_metrics.json`、`run_metadata`，历史目录的 `min_dist_same_elem` 已修复为 `null`（不再 NaN），日志可直接喂给 downstream `plot_eval/plot_compare`。

### 7.2 近期优先事项（对照 `todo_list`）
- Phase 1 晶格修复已落地：权威 gram6 表示、SPD+cond 投影、角度/cond softplus barrier、采样投影模式（every-step/final）与投影统计、fail_reason 标准化与 Top3 汇总、回归测试（grad/projection）已完成。
- 当前唯一未完成项仍是 **Phase 1 的可选任务**：针对层状材料对 MLIP（CHGNet 或同类）做小规模微调，并比较能量/力误差与 relax 成功率。
- 若启动该微调：先在 `P_TASK/data` 的 trajectory/force 子集上跑最小验证集对比（能量/力误差 + relax 成功率），确认收益后再替换采样/relax 的权重。

### 7.3 Phase 3 已完成项（2026-01-30）
- 采样侧新增 MLIP force guidance（CHGNet forces），支持在采样后段用 `--force-guidance` 注入能量梯度；采样配置会记录 guidance 参数。
- `prepare_c2db_tokens.py` 写入 `spacegroup_number/symbol`；训练与采样条件支持 `spacegroup` one-hot，并在 `eval_samples.py` 输出 `spacegroup_match/symmetry_violation` 与对应统计。
- `AtomDenoiser` 支持可选 `symmetry_loss_weight`（基于 spglib 的空间群不匹配惩罚）与 `tail_adapter=egnn`（最小等变 tail adapter）。
- 新增 `evaluate/io.py` 统一读取 `tier0/tier1/per_sample`（兼容旧命名），并加入 `evaluate/cache.py` 的 `energy_mlip`/`cross_vacuum` 缓存。
- 实现 `evaluate/self_train_loop.py`：采样 → 评估 → 筛选成功样本 → 生成 self-train 数据集（可合并 base npz）。
- 2026-02-05：对 `twodgen/` 下主要模块逐文件阅读后确认除 `problem.md` 记录的 Gram6 批处理瓶颈与 dataset `extra` 中对非数值的过滤外暂无新 blocker；同时把 `_parse_pbc_mask` 重复定义归类到 `tm.md`。

### 7.3 过程与记录
- 每次完成 `todo_list` 项或修复核心 bug 后，按 repo 指南同步更新 `twodgen/history.md`（记录时间+变更点）并在 `twodgen/process.md` 里标注新的进度节点，确保 `milps-plan` 与 `todo_list` 保持一致。
- 已补充 Phase2/3 未覆盖测试：新增 `tests/test_loss_schedule.py`（LossWeightScheduler 调度）、`tests/test_evaluate_io.py`（评估 IO 兼容旧命名）、`tests/test_symmetry_tail_adapter.py`（symmetry loss 回退与 EGNN tail adapter 掩码）。
- 补充 Phase2/3 剩余测试：新增 `tests/test_eval_cache.py`（evaluate/cache 能量与 cross_vacuum 缓存）。
- 追加 Phase2/3 训练侧约束测试：新增 `tests/test_atom_velocity_loss.py`，覆盖 vacuum loss 与 cross-vacuum loss 在 `AtomVelocityLoss` 中的输出与速率统计。

## 8) Phase 0-3 对齐（简述）
- **Phase0（基线 + 评估规范）**：`twodgen/evaluate/` 系列、`clean_c2db_2d` 的 quality bucket 与 `run_metadata` 使指标可复现；所有脚本都写 `run_metadata`。
- **Phase1（数据治理）**：canonical preprocess + quality tags + split(s) 已固化，`c2db_quality.jsonl` 供训练/采样/评估共享。
- **Phase2（有效率提升）**：`min_dist` penalty + curriculum、composition encoder 的 cond loss、cross-vacuum 评估、min_dist repulsion、vacuum loss 等措施均在训练/采样有日志。
- **Phase3（评估扩展）**：`plot_eval`, `plot_compare`, `run_pipeline` 为 Tier1/2/3 预留，目前 Tier0/1 输出固定格式，Tier2/3 正在补充 `property_predict`/formation-energy 等内容。

## 9) Code Review Notes (2026-01-31 / 2026-02-01)
- 复盘 `twodgen/scrip/sample_tokens.py`、`twodgen/evaluate/eval_samples.py` 等模块时发现 evaluation 假设 `cond_counts_vector` 始终存在，`cond_match_suspect` 判定会在缺失时触及未定义的 `exact_match`/`l1` 变量（详见 `problem.md`），导致无条件采样的 `eval_samples` 直接崩溃；已在 `eval_samples.py` 中增加空数组与条件分支，避免崩溃。
- `eval_samples.py` 已把 success 拆成 `success_geom/success_energy` 并写入 `energy_available`，避免缺失 `energy_mlip` 时 success 全部归零。
- 同次阅读里发现 `sample_tokens` 在文件顶部与 `parse_args()` 后各自定义了一份 `_parse_pbc_mask`，增添冗余，已经同步记录于 `tm.md`，可保留一处并删除另一份。
- 2026-02-05 项目梳理：按 `twodgen/` 目录逐文件过了一遍，除了 `problem.md` 中记录的几何投影性能与 dataset extra 数值/非数值区分的问题，暂未发现新代码层面 blocker；所有主要模块（data preprocess、dataset、train/sample/eval）都与现状同步并写有元数据/condition guard。

## 10) 最新训练与评估记录（2026-01-31 / 2026-02-01）

### 10.1 Run: `outputs/checkpoints/20260131_142943`
- checkpoint：`outputs/checkpoints/20260131_142943/atomdenoiser_best.pt`、`outputs/checkpoints/20260131_142943/atomdenoiser_last.pt`。
- 配置要点（来自 `config.json`）：
  - `cell_rep=cholesky6`，`g_scale=100.0`，`pbc_mask=(1,1,0)`；`pred_target=x0`。
  - 条件：`use_condition=True`，`cond_fields=[counts_vector,lattice_param,t]`。
  - 训练：`batch_size=256`，记录到 `train_metrics.jsonl`，该 run 最后一次记录 step 为 **17150**。
- 训练侧观测（聚合自 `train_metrics.jsonl`）：
  - `chol_log_clamp_rate` 均值约 **0.73**（p90 ~0.91），`chol_bound_rate` 均值约 **0.79**（p90 ~0.91），说明 Cholesky 对角 log 约束频繁触发。
  - `collision_rate` 均值约 **0.046**，`min_dist_mean` 均值约 **2.24**，`min_dist_p10` 均值约 **1.75**（该指标来自训练 batch 的真实结构，不代表采样质量）。
  - `loss_cond` 在该 run 的日志中始终为 0（未出现 `loss_cond_number` 字段），需要核对：cond/condition-number 约束是否实际参与 loss，以及 metrics 命名是否与当前代码一致。
  - `loss_angle` 在该 run 中大部分 step 为 0（偶尔非零），但采样评估显示角度问题非常严重，疑似 angle 约束未正确作用到“物理 lattice”（或作用点/缩放不一致）。
  - `lengths_std_mean` 在日志中出现 NaN（该字段作为健康度指标目前不可用，需修复统计方式）。

### 10.1.1 Run: `outputs/checkpoints/20260201_142022`（10-epoch short run）
- 配置要点（来自 `config.json`）：
  - `cond_max=1000.0`，`cond_loss_weight=0.01`，`loss_weight_warmup_steps=15000`（cond 约束处于 warmup 早期）。
  - `cell_rep=cholesky6`，`g_scale=100.0`，`pbc_mask=(1,1,0)`；`pred_target=x0`。
- 训练侧观测（聚合自 `train_metrics.jsonl`，step~0-1000）：
  - `loss_cond_number` 已按新字段写入，但始终为 0：`pred_cond_mean` 多在 20–60，远低于 `cond_max=1000`，因此未触发 condition-number hinge；同时 warmup 初期 `lambda_cond` 仍极小，短跑内几乎不生效。
  - `lengths_std_mean` 不再写入（NaN 已被过滤），日志更干净。
  - `loss_cross_vacuum`/`cross_vacuum_rate` 字段存在，说明跨真空惩罚在日志侧已能追踪。

### 10.2 Eval: `outputs/samples_tokens/eval`（200 samples, post-fix）
- Tier0（`tier0_metrics.json`）：
  - `valid_rate_eval=0.085`（约 17/200）。
  - 主要失败：`angle_out_of_range_rate=0.895`，并伴随 `collision`（fail_reason_counts: collision=51, angle_out_of_range=179）。
  - `cond`（lattice condition number）分布极端：mean ~1.93e6、median ~3.62e4，表明大量晶格病态/近奇异（与角度越界高度相关）。
  - `success_rate=0.08`，`success_geom_rate=0.08`，`energy_available_rate=0.0`；`energy_mlip.count=0`（缺能量时不再把 success 全部归零，几何成功率可读）。
- Tier1-2D（`tier1_2d_metrics.json`）：
  - `valid_2d_rate=0.08`。
  - `thickness` 与 `vacuum` 偏大（thickness mean ~58A，vacuum mean ~36A），尽管 `vacuum_ok_rate=0.76`，但整体几何尺度明显不合理。

### 10.3 当前暴露出的关键问题（按优先级）
- **晶格几何失真是主要瓶颈**：角度越界占比极高，且 lattice condition number 巨大，说明 cell 参数化/缩放/约束链路存在系统性问题（训练侧 angle/cond 约束未有效压住）。
- **cond 约束实际未触发**：当前 `cond_max=1000` 且 warmup 仍在早期，`pred_cond_mean` 远低于阈值，导致 `loss_cond_number` 恒为 0。需要调整阈值或 warmup 才能在训练中真正压制病态晶格。
- **能量链路缺失仍阻碍能量成功率**：评估已拆出 `success_geom/success_energy`，但 `energy_available_rate=0` 说明采样未计算 `energy_mlip`，仍无法比较“能量成功率”。
- **历史日志字段已对齐**：新 run 已统一 `loss_cond_number` 命名，`lengths_std_mean` NaN 已过滤，不再污染汇总。

### 10.4 下一步动作（建议）
- 先把“cond 约束真的生效”做成可验证闭环：降低 `cond_max` 或缩短 warmup，使 `loss_cond_number` 在训练中可触发，并持续记录 `pred_cond_mean` 与 `loss_cond_number` 的关系。
- 采样侧补上硬约束/投影：对角度、condition number、以及 in-plane 退化（近共线）做采样后投影或 clamp，避免大量样本在 eval 直接被 angle/cond 淘汰。
- 明确 relax/能量评估策略：在 `samples.npz`/`run_metadata` 中写入 `relax` 配置与依赖状态；评估侧把 success 拆成 `success_geom` 与 `success_energy` 两级，避免缺失 `energy_mlip` 时“全失败”掩盖几何改进。

### 10.5 E0 cond warmup 对照（2026-02-05）
- 配置：与 `outputs/debug_cond_trigger_warmup/20260205_163235` 相同（seed=123, batch=64, log_interval=10, cond_max=40, cond warmup 80 steps），仅将 `--cond-loss-weight` 设为 0，输出到 `outputs/debug_cond_trigger_warmup_off/20260205_170907`（因本机权限限制，num_workers=0，其他参数保持一致）。
- 触发性：on 组 `loss_cond_number` 均值 ~0.10、`pred_cond_mean` 均值 ~27.7；off 组 `loss_cond_number` 全程 0 且 `pred_cond_mean` 记录为 NaN，说明 cond 约束/统计被完全关闭（需要额外日志改造才能在 off 组继续观测 cond 数值）。
- 副作用：主 loss 分量变化不大（loss_f/on=3.80 vs off=3.49，loss_g/on=1.55 vs off=1.74，loss_z/on=4.67 vs off=4.67，loss_comp 持平），min_dist/collision 相同（collision_rate 仍为 0），训练仍能完整跑完 1 epoch。
- 2026-02-06：补齐 phase1b 剩余实现：训练端同时输出 cond_gram/cond_lattice 全套统计 + abs/rel diff + Spearman + valid_rate（与 cond loss 权重解耦）；新增 cond_max schedule（linear/cosine）与 `--debug-grad-submodules` 梯度探针；新增回归测试 `tests/test_cond_constraint.py` 覆盖 cond 触发与 NaN 防护；`debug_cond_trigger.sh` 保持 on/off 一键复现。

### 10.6 phase1b 验收闭环（2026-02-06）
- 训练口径修正：cond_gram 取平方根以匹配 cond_lattice（理论 cond(L)^2 ↦ sqrt），abs_diff_mean < 5e-6；A2 对齐。
- 短跑触发：`outputs/debug_cond_trigger/on_long_v2/20260205_222822` 早期 `loss_cond_number`>0，后期收敛 <0.01，cond_violation_rate→0；A1 满足。
- 采样对照：`outputs/samples_tokens/cond_on_fix`（cond loss=0.1, cond_max=40, project_gram_cond=40） vs `cond_off_baseline`（同配置 cond_loss_weight=0）：
  - cond_violation_rate 0.094 vs 0.203，cond_overflow 失败 6 vs 13，project_trigger_rate 0.22 vs 0.27。
  - 碰撞/角度失败率相近，说明改进主要集中在 cond；A3 联动下降已可观察。
- 能量闭环：采样侧启用 CHGNet relax 后 `energy_available_rate=1.0`（见 `outputs/samples_tokens/cond_on_relax`/`cond_off_relax`）；self-train loop 支持 `--select geom_energy` 做“几何+能量”二级筛选（demo: `outputs/self_train_demo_geom_energy_v4`）。
- 下一步：在更长步数/大样本上复查 valid_rate（目前 0.02~0.17 受 bad_volume 影响），考虑调小 g_scale 或增加 volume clamp 以避免体积爆炸。

### 10.7 phase2 MVP 进展（2026-02-06）
- 采样端新增 post-step 投影兜底：`twodgen/common/projection.py` + `twodgen/model/atom_denoiser.py` + `twodgen/scrip/sample_tokens.py` 的 `--post-project*` 开关，支持 angle/cond/inplane 组合护栏，并导出 per-sample 与 per-step 统计。
- 2D cond 口径统一：2D slab（pbc_mask=1,1,0）默认用 **in-plane Gram cond**（eval 与 post-step cond clamp 同口径），避免真空轴主导 cond。
- 增加 volume clamp 护栏（projection key=`volume`）：从训练 npz 自动读取 volume bounds（p1/p99），只缩放 in-plane 两个 lattice 向量以压制 `bad_volume`。
  - A/B 结果（`outputs/ab_proj_phase2_v8`，num=128/steps=50/seed=123）：A `success_geom_rate=0.133` → B `0.484`，`bad_volume` 98→47，`post_project_trigger_any_rate=0.742`；collision 19→20（对 `post_project_vol_scale_inplane<0.98` 的样本追加更强 repulsion：iters=`min_dist_iter+20`、strength=`1.5x`）。
- 评估侧补齐 in-plane 退化判定与失败原因主因优先级：`twodgen/evaluate/eval_samples.py` 输出 `fail_reason_geom`（固定优先级）+ `inplane_degen_rate` + `cond_lattice`。
- 能量链路 taxonomy：`eval_samples.py` 增加 `energy_skipped_reason` 与 `fail_reason_energy`，并输出跳过率/失败原因计数；采样目录写 `run_metadata.json` 与 `projection_stats.json`，便于解释 energy_available 与投影幅度。
- relax gate：`sample_tokens.py` 仅对几何成功（valid & non-cross-vacuum）样本尝试 CHGNet relax，避免算力浪费并使 `energy_skipped_reason=geom_fail` 更一致；能量闭环示例：`outputs/eval_energy_phase2/eval`。
- 回归脚本与测试：新增 `twodgen/scrip/sampling_projection_ab.sh`、`twodgen/scrip/eval_with_energy.sh`；新增 `tests/test_sampling_projection.py`、`tests/test_energy_chain.py`，并修复 `tests/test_c2db_clean_2d.py` 的旧 import；当前 `uv run pytest` 全通过。
- 规划状态回写：`twodgen/plan_next_sampling_energy_mlip.md` 已标注完成状态（A/B/D/E 完成，C 为可选后置）。

### 10.8 下一步建议（2026-02-06）
- 扩大回归规模：把 `outputs/ab_proj_phase2_v8` 的 num_samples 提到 512/2048，检查 `success_geom_rate/bad_volume/collision` 是否仍稳定（避免小样本偶然性）。
- 若能量闭环要用于筛选：建议在 `eval_with_energy.sh` 固定 `--relax-device` 与 relax 超参，并记录运行耗时（后续为 go/no-go 提供成本项）。
- Workstream C（可选）：已补齐 finetune 的 go/no-go 报告模板与 registry stub（见 `twodgen/mlip_finetune_report.md`、`twodgen/model_registry.json`），后续只需把训练脚本/数据 manifest 对齐并填表即可做决策。
