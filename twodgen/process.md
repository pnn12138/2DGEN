# twodgen 项目策略（当前对齐实现）

本文件描述当前 token 线的项目脉络与实现逻辑，覆盖预处理、扩散训练、去噪/采样、评估，以及与配置/脚本的对应关系。

---

## 0) 总览
- 主路线：token 扩散（`Z/F/g`）+ 条件扩散（`counts_vector` 可选 `lattice_param/t`）。
- A++ v3 预处理：生成 slab canonical 表示。
- 邻居策略：默认 2D PBC（`pbc_mask=1,1,0`）在线 kNN。

## 0.1) 端到端流程（实现逻辑）
1. **数据准备**：从 C2DB CSV/CIF 解析结构，生成 token 缓存 npz（`data/prepare_c2db_tokens.py`）。
2. **训练入口**：`scrip/train_tokens.py` 读取 npz，构建 DataLoader → AtomDenoiser。
3. **模型前向**：
   - `AtomDenoiser` 调用 `AtomVelocityLoss` 构造扩散/flow 的训练目标与损失。
   - `AtomTransformer` 接收 `Z/F/g + t + cond`，输出 `pred_v_f/pred_v_g/logits_z`。
4. **优化与日志**：AdamW + schedule/EMA（训练脚本），loss 细项记录到 `train_metrics.jsonl`。
5. **采样**：`scrip/sample_tokens.py` 使用 checkpoint 进行扩散采样，导出 `samples.npz` 与可选 CIF。
6. **评估**：`evaluate/eval_samples.py` 对样本做 Tier‑0/1 统计与合法性评估。

以上环节均以 `Z/F/g` token 表示为主，`lattice_param/t` 当前主要用于条件或缓存字段，不是主扩散变量。
## 当前进度摘要（同步更新）
- 已统一 PBC 为 slab 2D（`pbc_mask=1,1,0`），避免 z 方向假邻居。
- 条件扩散默认仅化学式（`counts_vector`），通过 `--cond-fields` 预留扩展（如 XRD）。
- 评估已支持 `--pbc-mask`，可与训练/采样的 slab 2D PBC 对齐。

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
- A++ v3 额外字段：`z_canon, uvz, uv_angle, u, v, z_norm, t, a_hat, b_hat, n, lattice_param, counts_vector, order_idx`。

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
- 否则：
  - `loss = loss_f + lambda_g*loss_g + lambda_z*loss_z`

---

## 3) 去噪模型（Transformer/DiT-style）
- 输入：`Z, F, g, atom_mask, t, cond`。
- [CELL] token 用晶胞表征；原子 token 用元素 + torus 编码坐标。
- 动态 kNN sparse attention（原子-原子），CELL 与原子全连接。
- attention bias：距离 RBF + 元素对嵌入。

### 3.1 条件注入（当前工程实现）
本节描述 `cond` 在模型中的具体注入路径（代码：`model/atom_transformer.py`）。

**条件向量构建（训练脚本）**
- 条件字段来自 batch：默认 `counts_vector`，可加 `lattice_param/t` 等（`--cond-fields`）。
- `counts_vector` 先按 `max_atoms` 归一化；连续字段可按 `--cond-normalize-fields` 做 z-score。
- 最终拼接为 `cond_vec`，形状 `(B, cond_dim)`。

**时间步注入**
- 扩散时间步 `t` 先做 sinusoidal embedding，再经 `time_mlp` 得到 `cond_time`（形状 `(B, D)`）。
- 若启用条件，`cond_vec` 经 `cond_mlp` 投影为 `(B, D)`。
- 当前策略为**相加融合**：`cond = cond_time + cond_mlp(cond_vec)`；若 `cond_vec` 为空则只用时间条件。

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
- 当前实现仅在训练阶段做 drop；采样阶段默认不做 cond/uncond 双前向。

---

## 4) 采样策略
- 采样过程中保持 `pbc_mask=1,1,0` 进行 MIC 距离计算。
- 默认仅在最终输出投影（`frac` wrap + lattice clip）。
- 可选每步投影：`--project-each-step`。

---

## 5) 评估策略（当前实现）
- 结构合法性：最小距离、体积范围、规约后 CIF 导出成功率。
- 统计与可视化：`evaluate/eval_samples.py` + `evaluate/plot_eval.py`。
- 建议输出：`test_fig/` 目录保存对比图与统计图。

---

## 6) 注意事项与默认值
- slab 推荐 `pbc_mask=1,1,0`。
- `t` 的几何影响：通过 slab 邻居图与 z_norm 参与距离体现，条件向量也包含 `t`。
