# twodgen 项目脉络（当前实现）

> 目标：把 token 轨道的端到端实现写清楚——数据如何打标整理、模型/训练/采样/评估如何串联，每一步都指向具体模块/函数，方便后续调试与验证。

---

## 0) 架构概览
- 主轴是 `Z/F/g` token diffusion：`Z`（原子序号）、`F`（fractional coords）、`g`（Gram6 或 cholesky6）分成一个序列，交给 `AtomTransformer` 进行跨原子 self-attn，丁点细节由 `AtomDenoiser` 封装。
- 所有几何变量以 canonical frame（`coord_frame="canon"`）和 align_atoms 保障可比较的 per-atom 顺序；`prepare_c2db_tokens.py` 选项 `--align-atoms` / `--coord-frame canon` 控制是否写入 `order_idx/order_inv`。
- 条件默认是 `counts_vector,lattice_param,t`（训练脚本`cond_fields`/`cond_normalize_fields`），并走 composition encoder 为元素计数提供额外 embedding。
- slab 2D PBC（`pbc_mask=(1,1,0)`）贯穿数据、训练、采样与评估，任何需要 3 层 2D 物理的逻辑都会显式参考 `twodgen/common/crystal.py` 提供的 `frac_mic_dist`、`gram6<->cholesky`、`build_knn` 等工具。

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

### 4.2 采样监控
- `samples.npz` 记录 `chol_log_clamp_rate`（`AtomTransformer` 中 clamp 触发统计），`min_dist_before/after` 由 `eval_samples` 使用 `frac_mic_dist` 统一计算。
- `outputs/samples_tokens` 结构里有 `samples.npz`, `eval`, `logs` 目录，`eval_samples` 还在 `per_sample.jsonl` 中写 `cond_matched` 等 200 条样本的详细字段。

---

## 5) 评估与指标

### 5.1 `evaluate/eval_samples.py`
- 对 `samples.npz` 逐行复现 `AtomDenoiser` 的 `frac`, `lattice`，用 `_min_dist_and_shifts` 计算 exact MIC min_dist，并按 `min_dist_cut=1.5`/`bond_cut`/`dup_eps` 做过滤。
- Tier 0：`valid_rate_eval`、`valid_criteria`、`min_dist/min_dist_collision/min_dist_same_elem`、`volume/cond/angle` 分布；Tier 1：`valid_2d_rate`, `thickness`, `vacuum`, `cross_vacuum_rate`, `vacuum_ok_rate`。
- `per_sample.jsonl` 记录 `valid`, `valid_2d`, `fail_reason`, `cond_exact_match`, `cond_l1`, `vacuum`, `cross_vacuum_bond`，没有重复元素时 `min_dist_same_elem=null`，避免 NaN；旧的 `per_sanmple.jsonl`/`tier*_metric.jsonl` 在写入时自动改名（手动 rename 已统一），确保历史目录与当前输出一致。
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
- Phase 1 Fixer、Phase 2 训练约束、Phase 3 架构升级、评估缓存/闭环已完成并落地；当前唯一未完成项是 **Phase 1 的可选任务**：针对层状材料对 MLIP（CHGNet 或同类）做小规模微调，并比较能量/力误差与 relax 成功率。
- 若启动该微调：先在 `P_TASK/data` 的 trajectory/force 子集上跑最小验证集对比（能量/力误差 + relax 成功率），确认收益后再替换采样/relax 的权重。

### 7.3 Phase 3 已完成项（2026-01-30）
- 采样侧新增 MLIP force guidance（CHGNet forces），支持在采样后段用 `--force-guidance` 注入能量梯度；采样配置会记录 guidance 参数。
- `prepare_c2db_tokens.py` 写入 `spacegroup_number/symbol`；训练与采样条件支持 `spacegroup` one-hot，并在 `eval_samples.py` 输出 `spacegroup_match/symmetry_violation` 与对应统计。
- `AtomDenoiser` 支持可选 `symmetry_loss_weight`（基于 spglib 的空间群不匹配惩罚）与 `tail_adapter=egnn`（最小等变 tail adapter）。
- 新增 `evaluate/io.py` 统一读取 `tier0/tier1/per_sample`（兼容旧命名），并加入 `evaluate/cache.py` 的 `energy_mlip`/`cross_vacuum` 缓存。
- 实现 `evaluate/self_train_loop.py`：采样 → 评估 → 筛选成功样本 → 生成 self-train 数据集（可合并 base npz）。

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
