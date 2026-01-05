# twodgen 项目策略（当前对齐实现）

本文件描述当前 token 线的项目脉络与实现逻辑，覆盖预处理、扩散训练、去噪/采样、评估，以及与配置/脚本的对应关系。

---

## 0) 总览
- 主路线：token 扩散（`Z/F/g`）+ 条件扩散（`counts_vector` 可选 `lattice_param/t`）。
- `g` 表示晶胞 token：支持 `gram6` 与 `cholesky6`（内部以 `g_scale` 归一化）。
- A++ v3 预处理：生成 slab canonical 表示。
- 邻居策略：默认 2D PBC（`pbc_mask=1,1,0`）在线 kNN；采样/评估可覆盖。

## 0.1) 端到端流程（实现逻辑）
1. **数据准备**：从 C2DB CSV/CIF 解析结构，生成 token 缓存 npz（`data/prepare_c2db_tokens.py`）。
2. **训练入口**：`scrip/train_tokens.py` 读取 npz，构建 DataLoader → AtomDenoiser。
3. **模型前向**：
   - `AtomDenoiser` 调用 `AtomVelocityLoss` 构造扩散/flow 的训练目标与损失。
   - `AtomTransformer` 接收 `Z/F/g + t + cond`，输出 `pred_v_f/pred_v_g/logits_z`。
4. **优化与日志**：AdamW + schedule/EMA（训练脚本），loss 细项记录到 `train_metrics.jsonl`。
5. **Checkpoint**：保存 `model_state_dict`、可选 `ema_state_dict`、`config`（模型）、`diffusion_config`、`cond_config`。
6. **采样**：`scrip/sample_tokens.py` 使用 checkpoint 进行扩散采样，导出 `samples.npz` 与可选 CIF；可选 `--eval` 直接输出评估结果。
7. **评估**：`evaluate/eval_samples.py` 对样本做 Tier‑0/1 统计与合法性评估，并在条件采样时计算化学式匹配指标。

以上环节均以 `Z/F/g` token 表示为主，`lattice_param/t` 作为辅助几何变量进入训练并可在采样时更新，但不是主 token。
## 当前进度摘要（同步更新）
- 已统一 PBC 为 slab 2D（`pbc_mask=1,1,0`），避免 z 方向假邻居。
- 条件扩散默认仅化学式（`counts_vector`），通过 `--cond-fields` 预留扩展（如 XRD）。
- 几何头已接入训练：`uv_angle/z_norm/lattice_param`（可选 `t` 厚度）。
- 采样支持几何更新：`--project-geometry` 会更新 `uv_angle/z_norm/lattice_param/t`，并仅对 `uv_angle/z_norm` 做投影。
- 双图与 wrap embedding 已接入：`--dual-graph` + `--wrap-embed-dim`。
- 评估已支持 `--pbc-mask`，可与训练/采样的 slab 2D PBC 对齐。
- 采样可写入条件计数向量（`cond_counts_vector`），评估会输出化学式匹配指标。
- canonical 坐标系已打通：缓存支持 `f_canon/lattice_canon/gram6_canon`，训练/采样默认 `--coord-frame canon` 并对齐 per-atom 字段。
- 评估脚本已去掉采样循环依赖；plot_compare 默认精确 MIC（可用 `--mic-mode approx` 复现旧图）。

---

## 1) 预处理与缓存（A++ v3）
目标：把 slab 几何规约到可复现、可缓存的输入，训练端避免重复做 reduction/投影。

### 1.1 输入
- CIF 解析得到 `cell + pos_cart + Z`（单位晶胞原子列表）。

### 1.2 核心步骤（A++ v3）
- slab 法向 `n`（`a×b`，退化时 PCA fallback）。
- 投影到面内/面外，得到 `r_parallel` 与 `z`。
- 2D reduction 得 `a_hat, b_hat`（unimodular 枚举）。
- 计算 `u,v`（在 `a_hat,b_hat` 基下的分数坐标）并 wrap。
- torus mean shift 去面内整体平移 gauge。
- `z` 平移到均值 0，翻转 slab 方向（`m1/m3` 判别）。
- 厚度 `t` 与 `z_norm` 归一化（并 clip）。
- `lattice_param`（log-area + shape，log-Cholesky）。
- canonical 排序（`Z, z_norm, u, v`）。

### 1.3 缓存字段
- 训练主输入：`z, f, gram6, atom_mask`。
- canonical 追加字段：`f_canon, lattice_canon, gram6_canon, order_inv, schema_version, coord_frame`。
- A++ v3 额外字段：`z_canon, uvz, uv_angle, u, v, z_norm, t, a_hat, b_hat, n, lattice_param, counts_vector, order_idx, u_shift, v_shift`。

### 1.4 Token 缓存格式与约定
- 由 `data/prepare_c2db_tokens.py` 写出的 npz 需包含 `gram6_convention="row_lattice"`。
- `C2DBTokenNPZDataset` 会检查 `gram6_convention`，不满足即报错并要求迁移。
- 晶胞向量约定：行向量 lattice（`cart = frac @ lattice`）。
- `gram6 = lattice_to_gram6(lattice) / g_scale`，训练/采样内部始终在 scaled 空间。
- `counts_vector` 为元素计数（长度 118，Z 从 1 起）；采样/训练时会按 `max_atoms` 归一化。

### 1.5 原始 CSV 数据集路径
- `C2DBAtomDataset` 直接从 `c2db_summary.csv` 解析 CIF 并 padding。
- 可选 `--niggli-reduce` 在读取时做 Niggli reduction；失败则回退原结构。
- 输出字段与 npz 一致，保持 `atomic_numbers/frac_coords/atom_mask/gram6` 对齐。

---

## 2) 扩散训练策略

### 2.1 输入与条件
- 原子 token：`Z`（离散）+ `F`（分数坐标）+ `g`（Gram6 / Cholesky6）。
- 条件向量（可选）：`counts_vector` + `lattice_param` + `t`（归一化后拼接，`t` 为 slab 厚度，不是扩散时间步）。

### 2.2 邻居与注意力
- 默认在线 kNN（`frac_mic_dist`）：使用 `pbc_mask=1,1,0` 仅面内周期，z 非周期。
 

### 2.3 扩散目标
- `F`/`g`：v‑pred 回归（连续扩散）。
- `Z`：mask diffusion（离散分类）。

### 2.4 Loss 形式（当前实现）
实现位置：`common/atom_diffusion.py::AtomVelocityLoss`

**时间采样**
- diffusion 模式：`t ~ sigmoid(N(P_mean, P_std))`
- flow 模式：`t ~ Uniform(0, 1)`

**连续分支（F 与 g）**
- 生成噪声：`noise_f ~ N(0, I)`，`noise_g ~ N(0, I)`
- diffusion 模式：
  - `frac_t = t * frac + (1 - t) * noise_f`
  - `cell_t = t * cell + (1 - t) * noise_g`
  - `v_f = (frac - frac_t) / (1 - t)`
  - `v_g = (cell - cell_t) / (1 - t)`
- flow 模式：
  - `frac_t = t * noise_f + (1 - t) * frac`
  - `cell_t = t * noise_g + (1 - t) * cell`
  - `v_f = noise_f - frac`
  - `v_g = noise_g - cell`
- 损失：
  - `loss_f = MSE(pred_v_f, v_f)`（按 `atom_mask` 平均）
  - `loss_g = MSE(pred_v_g, v_g)`（cell 级别）

**离散分支（Z）**
- mask 比例：`p_mask = schedule(t)`（flow: 线性增；diffusion: 线性减）
- 随机 mask：将部分 `z` 替换为 `mask_id`
- 损失：
  - `loss_z = CE(logits_z[masked], z[masked])`

**总损失**
- 若 `use_uncertainty_weighting=True`（默认）：
  - `loss = exp(-s_f)*loss_f + s_f + exp(-s_g)*loss_g + s_g + (exp(-s_z)*loss_z + s_z)`
  - `s_f/s_g/s_z` 为可学习标量
  - 若启用几何头，会额外加入 `loss_uv/loss_zn/loss_lat/loss_t` 及对应的 `s_*` 项
- 否则：
  - `loss = loss_f + lambda_g*loss_g + lambda_z*loss_z`
  - 若启用几何头，会额外加上 `lambda_uv/lambda_zn/lambda_lat/lambda_t` 项
  - `loss_t` 只在数据包含 `t` 且启用几何分支时出现；不需要厚度预测可关闭几何分支

### 2.5 训练脚本要点（实现细节）
- `train_tokens.py` 根据 `--use-condition/--cond-fields/--cond-normalize-fields` 推导 `cond_dim` 与 `cond_stats`。
- `counts_vector` 在训练和采样中均按 `max_atoms` 归一化。
- `cond_stats` 会序列化进 `cond_config`；采样时若缺失会从 `cond-npz` 重新估计。
- `cond_drop_prob` 仅在训练阶段生效（CFG-style），采样默认仅条件前向。
- 数据加载：
  - `C2DBTokenNPZDataset` 直接读取缓存，`C2DBAtomDataset` 从 CSV 解析 CIF。
  - `--bucket-batches` 使用 `BucketBatchSampler` 按原子数排序后分桶，减少 padding。
  - worker 级 seed 固定，保证可复现实验。
- 学习率：
  - 线性 warmup + cosine 或 constant；每 step 调整。
  - `--clip-grad` 对所有参数做 `clip_grad_norm_`。
- EMA：
  - `--ema` 启用指数滑动平均；权重保存至 `ema_state_dict`。
  - 采样时 `--use-ema` 优先加载 EMA。

---

## 3) 去噪模型（Transformer/DiT-style）
- 输入：`Z, F, g, atom_mask, t, cond`；若启用 composition encoder 还会接收 `counts_vector`。
- [CELL] token 用晶胞表征；原子 token 用元素 + torus 编码坐标。
- 动态 kNN sparse attention（原子-原子），CELL 与原子全连接。
- attention bias：距离 RBF + 元素对嵌入。

### 3.1 条件注入（当前工程实现）
本节描述 `cond` 在模型中的具体注入路径（代码：`model/atom_transformer.py`）。

**条件向量构建（训练脚本）**
- 条件字段来自 batch：默认 `counts_vector`，可加 `lattice_param/t` 等（`--cond-fields`）。
- `counts_vector` 先按 `max_atoms` 归一化；连续字段可按 `--cond-normalize-fields` 做 z-score。
- 最终拼接为 `cond_vec`，形状 `(B, cond_dim)`。

**composition encoder（模型内部）**
- 若 `use_comp_encoder=True` 且提供 `counts_vector`，模型内部会做元素嵌入 + pooling，得到 `cond_comp`。
- `cond_comp` 通过标量门控与 `cond_time/cond_vec` 融合：`cond = w_time*cond_time + w_vec*cond_vec_proj + w_comp*cond_comp`。
- `w_comp` 默认从 0 起步，保证兼容旧 checkpoint 并防止早期训练不稳。

**时间步注入**
- 扩散时间步 `t` 先做 sinusoidal embedding，再经 `time_mlp` 得到 `cond_time`（形状 `(B, D)`）。
- 若启用条件，`cond_vec` 经 `cond_mlp` 投影为 `(B, D)`。
- 当前策略为**门控相加融合**：`cond = w_time*cond_time + w_vec*cond_mlp(cond_vec) + w_comp*cond_comp`；若 `cond_vec` 为空则只用时间条件。

**在 Transformer block 中的 AdaLN 调制**
- 每个 block 内部用 `cond` 生成 6 路参数：`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`。
- 这些参数用于调制 LayerNorm 输出并门控 residual：
  - MSA 路径：`x = x + gate_msa * Attn(modulate(LN(x)))`
  - MLP 路径：`x = x + gate_mlp * MLP(modulate(LN(x)))`
- 该注入方式等价于 DiT 风格 AdaLN + gating，条件对每层均可见。

**条件参与 atom-cell 交互 bias**
- 若启用 `cell_bias`，会构造 `bias_atom_cell` 与 `bias_cell_atom`：
  - `cond` 与 atom token / cell token 拼接后经 MLP 得到 per-head bias。
  - 该 bias 只作用于 atom<->cell 的 attention logits。

**条件 dropout（CFG-style 训练）**
- 训练中 `cond_drop_prob` 可将 `cond_vec` 置零形成“无条件”样本。
- 若启用 composition encoder，会同步把 `counts_vector` 置零，保证 unconditional 分支真实。
- 当前实现仅在训练阶段做 drop；采样阶段默认不做 cond/uncond 双前向。

---

## 4) 采样策略
### 4.1 采样配置与加载
- 采样读取 checkpoint 中的 `config/diffusion_config/cond_config`，缺失字段使用脚本默认值。
- `--npz` 用于统计 `num_atoms` 分布与体积范围（`v_min/v_max`），也可作为默认 `cond-npz`。
- `--pbc-mask` 覆盖模型默认；若未指定且 config 缺失，使用 `1,1,0`。

### 4.2 晶胞初始化与投影
- `cell_rep=cholesky6` 时，若有 `--npz` 会用体积统计推导 `cell_init_scale`、`cell_init_noise`、`chol_log_min/max`。
- `--cell-init iso` 启用各向同性先验；`--cell-init-scale`/`--cell-init-noise` 可显式覆盖。
- 默认仅在最终输出投影（`frac` wrap + lattice clip）；`--project-each-step` 会在每步执行投影并启用 `clip_lattice`。

### 4.3 条件与原子数
- 若 `cond_config.use_condition=True`，采样必须提供 `--cond-npz`（或 `--npz` 兼作）。
- `cond_fields` 从 checkpoint 的 `cond_config.cond_fields` 推断；若为空则回退到 `counts_vector`（可选 `lattice_param/t`）。
- 条件采样时若提供 `counts_vector`，采样原子数 `N` 由 `counts_vector.sum()` 决定；否则从 `--npz` 的原子数分布随机采样，或用 `--num-atoms` 固定。

### 4.4 采样输出
- 生成 `samples.npz`（含 `z/frac/lattice/atom_mask`，以及可选 `cond_indices/cond_counts_vector`；若启用几何采样会额外写入 `lattice_param/t`）。
- CIF 输出由 `--save-cif/--cif-mode/--cif-filter` 控制。
- `--eval` 可立即生成 `eval/per_sample.jsonl` 与 Tier‑0/1 汇总。

### 4.5 采样内循环（实现细节）
- `AtomDenoiser.generate`：
  - 初始化 `z` 为 `mask_id`，`frac` 与 `cell` 采样高斯噪声。
  - 若启用 `--project-geometry`，同时初始化并更新 `uv_angle/z_norm/lattice_param/t`。
  - `diffusion` 模式：`t` 从 0→1；`flow` 模式：`t` 从 1→0。
- Euler/Heun：
  - `euler` 一步预测更新 `frac/gram6`。
  - `heun` 两次预测做修正（平均速度），更稳但慢。
- 离散 `Z` 采样：
  - 支持 `argmax/temperature/topk/topp`。
  - `mask_id` 通道被置为 `-inf`，避免采回 mask token。
- 邻居更新：
  - `neighbor_update_steps` 控制 kNN 更新频率，减少开销。
  - MIC 距离用 `model_cfg.pbc_mask`。

---

## 5) 评估策略（当前实现）
- 结构合法性：最小距离、体积范围、规约后 CIF 导出成功率。
- 条件匹配：输出 `exact_match_rate`、计数误差与组成相似度。
- 统计与可视化：`evaluate/eval_samples.py` + `evaluate/plot_eval.py`。
- 建议输出：`test_fig/` 目录保存对比图与统计图。

### 5.1 评估产物
- `per_sample.jsonl`：每个样本的合法性、最小距离、体积、条件匹配等。
- `tier0.json`：整体统计与通过率汇总。
- `tier1.json`：分项统计（条件匹配、距离/体积等）。

---

## 6) 注意事项与默认值
- slab 推荐 `pbc_mask=1,1,0`。
- `t` 的几何影响：通过 slab 邻居图与 z_norm 参与距离体现，条件向量也包含 `t`。
