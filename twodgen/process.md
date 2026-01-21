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
- `prepare_c2db_tokens` 每行 CIF 走 `row_to_tokens`：先 pad（`_pad_1d/_pad_2d/_pad_2d4`）/mask；再用 `_mic_dist_and_shifts` 计算 `min_dist`、`cross_vacuum_bond`、`thickness`/`vacuum`；如果 `--preprocess-v3`，会调用 `preprocess_cartesian`、生成 canonical `z`, `f`, `gram6`, `lattice`，并把 canonical per-atom fields 写进 npz（`f_canon`, `atom_mask_canon`, `gram6_canon`, `uv_angle`, `uvz`, `z_norm`, `lattice_param`）。
- 最终 `np.savez` 包含 `z,f,atom_mask,gram6,lattice`、canonical 版本、`counts_vector`, `slab_thickness/vacuum`, `min_dist/collision_risk`、`schema_version`, `coord_frame`, `g_scale`, `max_atoms` 等元数据，`npz` 里的 `gram6_convention="row_lattice"` 保证 `g @ g^T` 变换兼容。

### 1.3 数据集类与切分
- `twodgen/data/c2db_dataset.py` 提供：`C2DBDataset`（从 CSV 解析 CIF，按 `pad_value` 填充 `atomic_numbers/frac_coords/lattice_matrix`）→ `C2DBAtomDataset`（在 `__getitem__` 中计算 Gram6 / `counts_vector`、可选 `niggli_reduce`）→ `C2DBTokenNPZDataset`（直接读 npz，按 `coord_frame` 选择 `raw`/`canon`，`align_atoms` 受 `order_idx` 约束）。
- `C2DBTokenNPZDataset` 也会把 `extra` 字段（只要 shape[0] = n_rows）缓存进 `dataset.extra`，供 `train_tokens._compute_dataset_min_dist` 直接重用 `min_dist`，避免重算 `frac_mic_dist`。
- `IndexedDataset` wrap 任何 base dataset，加一个 `index` 字段供训练/日志使用； `IndexedDataset.collate_fn` 特殊处理 `index`，`_unwrap_indexed_dataset` 在 `_compute_dataset_min_dist` 中用来跳过该字段，防止 `torch.stack` 报错。
- 划分用 `twodgen/data/create_c2db_split.py` 按 `n_atoms`, `element counts`、`vacuum` 等做 stratified split，生成 `trainsplit/heldout` JSON；训练脚本可用 `--split-json/--split` 指定 `train`/`heldout` 子集，再通过 `_filter_indices_by_quality` 结合 `c2db_quality.jsonl` 的 `quality_bucket` 做 `hard_pass`/`risk` 过滤（`quality_map` 由 `_load_quality_map` 读取）。

### 1.4 统计与可视化工具
- `twodgen/data/splits.py` 里封装 `select_split_indices/load_c2db_split/validate_split_indices`，`twodgen/data/torus.py::torus_encode` 提供 `uv` embedding； `twodgen/data/dataset.py`（CrystDataset）为少数 legacy 工程保留 `C2DBTokenNPZDataset` alias。
- `twodgen/data/clean_c2db_2d.py` 内的 `_thickness_vacuum`、`_mic_dist_and_shifts` 也会在训练/采样中复用（通过 `twodgen/data/prepare_c2db_tokens.py` 的导入），确保 metrics 一致。

---

## 2) 训练管线

### 2.1 入口与配置（`twodgen/scrip/train_tokens.py`）
- CLI 解析大量参数：`--npz`/`--csv` 选择数据源，`--align-atoms`/`--coord-frame` 控制 canonical frame，`--min-dist-train-cut/--min-dist-train-weight` 配置碰撞 penalty，`--curriculum-collision`/`--filter-min-dist-below` 控制样本在线筛选，`--vacuum-loss-weight/--vacuum-min` 绑定 vacuum 惩罚。
- 数据集构建：`dataset = C2DBTokenNPZDataset(...) if --npz else C2DBAtomDataset(...)`；无论哪种数据源，都会 wrap 成 `IndexedDataset` 方便日志追踪；base dataset 的 `g_scale`、`coord_frame_actual` 等 metadata 直接注入日志/警告。
- 训练前预处理：质量过滤（`quality_map`）、curriculum indices（`_curriculum_indices`）根据 `min_dist_all`（`_compute_dataset_min_dist` 支持直接用 cached + `frac_mic_dist`）决定 sample order；`_normalize_cond_stats`、`collect_run_metadata` 记录 `cond_stats`、`run` ID，训练开始便写入 `outputs/checkpoints/<run>/config.json`。
- `train_one_epoch` 每 step：从 loader 得 `z`, `frac`, `atom_mask`, `gram6`，同时 `batch_indices`（index）用于在 `min_dist_inf` 记录异常样本；每 `log_interval` 计算 `chol_log_clamp_rate`（通过 `gram6_to_cholesky6` 判断是否 hit log bounds）、`min_dist_mean/p10/collision`、`loss_uv/loss_zn/loss_lat` 等 metrics，追加写入 `train_metrics.jsonl`，还有 `train_metrics/config.jsonl`＋`run_metadata`。
- 参数优化器使用 `AdamW` + cosine LR schedule（`_compute_lr/_adjust_lr`），EMA 选项可通过 `--ema` 开启；`save_checkpoint` 负责写 `atomdenoiser_last/best.pt`，`model_config`, `diffusion_config`, `cond_config` 三份配置一同持久化。

### 2.2 losses & constraints
- 基础 diffusion loss：`AtomDenoiser._predict_velocity` 目标 `x0`，通过 `AtomVelocityLoss`（`common/atom_diffusion.py`）把 `x0` 转成 velocity、Gram residual、`Z` logit；`z` 走 mask diffusion（`logits_z`）并结合 `ignore_tokens`.
- Geometry heads：`uv_angle`, `z_norm`, `lattice_param`, `t`（thickness）在 `AtomDenoiser._project_geometry_step` 被 `atom_mask` 约束；`project_each_step` 控制是否在每个采样 step 投影 `frac/gram6`，训练中 `use_geometry_fields` 依赖 dataset 是否有 canonical fields。
- Collisions/vacuum：训练有 `min_dist` penalty（`min_dist_train_weight`），只惩罚小于 `min_dist_cut` 的 pairs；`vacuum_loss_weight` 通过 `AtomDenoiser._heun_step` 里隐含 `slab_t` 预测对 `0`/`1` 的 ramp ，log 输出 `loss_vacuum`。
- `chol_log_relax`/`chol_log_min/max` 在 `AtomDenoiser._project_step` 中控制 cholesky clamp，`AtomTransformer` 会在 `cholesky6_to_lattice` 与 `lattice_to_gram6` 之间加 clamp；`chol_log_clamp_rate` 统计 clamp 命中率，帮助诊断 lattice collapse。
- 条件 loss：`cond` 由 `counts_vector`/`lattice_param`/`t`（可能加 `cond_drop_prob`），再加 composition encoder（`AtomTransformerConfig.use_comp_encoder`）产生 `cond_comp`；通过 `lambda_cond`、`comp_loss_weight`（l1/cos）控制 `loss_cond`/`loss_comp`。
- 物理约束调度：增加了 `twodgen/loss/schedule.py::LossWeightScheduler`，默认 warm-up 3w 步、支持 linear/sigmoid/cosine 三类曲线，把 `vacuum/cond/chol_bound/expand_collision/volume/c_len` 等物理项纳入统一调度；`train_tokens.py` CLI 新增 `--volume-loss-weight`/`--c-len-loss-weight`/`--loss-weight-schedule` 等参数，训练配置会持续向 `train_metrics.jsonl` 输出 `loss_volume/loss_c_len`、`loss_chol_bound`、`lambda_*` 等指标，便于对比不同调度方式的效果。
- 全局 lattice 约束：在 `twodgen/common/atom_diffusion.py` 里新增 volume/c_len penalty，结合已有的 `loss_chol_bound` 与 expand-on-collision，通过 `lambda_volume`/`lambda_c_len`、`c_len_min/vacuum_min` 等参数强制生成的晶胞满足面积/体积和 c 轴真空标准；expand-on-collision 惩罚现在根据距离严重度放大，并且 `LossWeightScheduler` 会在训练早期就让这些约束逐步释放，不再等到后期才“补救” collapse。

### 2.3 数据增强与后处理
- `min_dist` repulsion在训练/采样都可启用：`AtomDenoiser._apply_min_dist_repulsion` 迭代 `min_dist_iter` 次，`strength` 控制 push 散度，`frac` 需要 center-of-mass 修正避免漂移。
- 训练曲线写盘：`train_metrics.jsonl` 以 JSON lines 按 `log_interval` 写入 `loss/min_dist/chol_log_clamp/geometry losses`，还有 `per-step` stats（`min_dist_inf`, `min_dist_low_atoms` + sample indices）。
- 所有 cond/geometry stats 通过 `_normalize_cond_stats` 记录 `cond_stats`/`cond_fields`/`t_stats`，便于 later normalization（例如 `t` 只用训练 split 的 mean/std，避免泄漏）。

---

## 3) 模型结构详解

### 3.1 AtomTransformer 主干
- `AtomTransformer` 接受 concat [cell token + atoms]，Cell token 与 atoms 共享 embedding；`GatherAttention` 分别计算 cell→atoms、atoms→neighbors 的 attention，`bias_nbr`/`bias_atom_cell`/`bias_cell_atom` 由 `build_knn` 提供 neighbor bias/atom-cell bias/graph bias。
- 位置编码：`torus_encode` 和 `rbf_expand` 为 fractional coords/Gram distances 编码；`modulate` 结合 `cond` 生成 AdaLN 风格的 shift/scale/gate，`AtomBlock` 包含 attention + MLP + gating 组成。
- 结构上支持 `dual_graph`、`wrap_embed_dim`、`edge_type_gating`、`cache_neighbors`、`edge_type_dim`、`element_ids` 等开关，allow customizing kNN or gating for ablations。

### 3.2 AtomDenoiser 包装
- 在 `twodgen/model/atom_denoiser.py` 中，`AtomDenoiser` 先把 `frac`/`gram6`/`z` embed 进 `AtomTransformer`，再由 `AtomTransformer` 的 output 拆成 `vel`、`uv_angle`、`z_norm`、`lattice_param`、`t`（`geom_preds`）。
- `project_geometry_step` 使 `uv_angle`/`z_norm` 归一；`_project_step` 会对 `gram6` 做 cholesky clamp，投影回 `frac`/`cell`，并调用 `clip_lattice` 维护 `volume/cond` 在 `[v_min,v_max]`、`cond_max` 内。
- `loss_min_dist`、`loss_angle`、`loss_cond` 通过 `metrics` dict 和 training loop 进行 logging；`pred_cond_mean`/`pred_angle_out_rate` 记录 geometry 侧状态。`_predict_velocity` 也可能返回 `_apply_min_dist_repulsion` 之后的 frac。
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
- `property_predict.py` 目前是 placeholder，用于 Tier-2 未来引入 formation energy 评估；`evaluate/check_conditions.py`/`formation_energy.py` 仍在整理中。

### 5.3 指标、日志与历史
- 训练/采样都写 `train_metrics.jsonl`/`per_sample.jsonl`，`train_metrics` 目录里还存在 `config.jsonl`, `train_metrics.jsonl` 记录 `min_dist`, `collision_rate`, `chol_log_clamp_rate`、`cond_mean` 等 line-by-line metrics；历史日志里我们已把旧 typo 改名，保证脚本 `/train_metrics/tier0_metrics.json` 找到正确文件。

---

## 6) 当前进度与待办
- 预处理 & canonical：`prepare_c2db_tokens` 生成的 npz 含 `f_canon`, `atom_mask_canon`, `gram6_canon`, `uv_angle`, `z_norm`, `lattice_param`，训练时 `coord_frame="canon"`，geometry heads 可完全启用。
- 训练/采样的指标流水线已打通：`outputs/checkpoints/<run>/train_metrics.jsonl`、`train_metrics/config.jsonl`、`per_sample.jsonl`、`tier0_metrics.json`/`tier1_2d_metrics.json` 都已生成并记录 `run_metadata`，历史目录也同步 rename；`min_dist_same_elem` 不再输出 NaN 。
- 仍需针对高优先级问题进一步调参：`chol_log_clamp_rate` 仍有 0.3–0.96 的波动，Tier0 valid rate ≈ 6%、collision/ vacuum 仍没达到目标。下一步是尝试 `chol_log_relax`、减弱 clamp、强化 vacuum/collision loss，待新一轮完整训练后再复查指标。
- 下一步代码侧优先事项已在 `twodgen/todo_list.md` 补齐：loss 动态权重调度（warm-up/不确定性权重）、训练诊断仪表板（分布+梯度）、对称性硬约束（映射标签+residual loss）、可选等变图层（EGNN/GVP/TFN）、评估缓存/标签传播、形成能重排序与统一 `success_rate` 指标；并额外补充 P2：EGNN Tail adapter/对比（GVP/TFN）、组合条件重参数化与 FiLM 注入、stable 子集 fine-tune 与自回流训练闭环。
- 已补齐硬约束修复相关的训练端支持：新增 `loss_chol_bound` 与 expand-on-collision 训练 loss、loss 权重 warmup 调度、以及 `cond_match` 目标条件来源标记与 sanity check，便于后续将 valid rate 拉升到可用区间。

---

## 7) Phase 0-3 对齐（简述）
- **Phase0（基线 + 评估规范）**：`twodgen/evaluate/` 系列、`clean_c2db_2d` 的 quality bucket 与 `run_metadata` 使指标可复现；所有脚本都写 `run_metadata`.
- **Phase1（数据治理）**：canonical preprocess + quality tags + split(s) 已固化，`c2db_quality.jsonl` 供训练/采样/评估共享。
- **Phase2（有效率提升）**：`min_dist` penalty + curriculum、composition encoder 的 cond loss、cross-vacuum 评估、min_dist repulsion、vacuum loss 等措施均在训练/采样有日志。
- **Phase3（评估扩展）**：`plot_eval`, `plot_compare`, `run_pipeline` 为 Tier1/2/3 预留，目前 Tier0/1 输出固定格式，Tier2/3 正在补充 `property_predict`/`formation_energy` 等内容。
