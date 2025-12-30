# Transformer Crystal Diffusion 实现指南（推荐单一路线）

> 目标：以 **Transformer/DiT-style** 为主体，在扩散框架下生成晶体结构（元素种类 + 原子坐标 + 晶胞），并通过 **动态邻居（由坐标+晶胞推导）** 将“键合/局域相互作用”注入到注意力中。
>
> 你已确定：采用 **方案 A：[CELL] 全局 token**，不使用 SCDM 那种对 L 的 repeat。

---

## 0. 路线总览（你将实现的唯一版本）

### 关键决策（已固定）

- **输入表示**：每个原子一个 token，序列长度 `N` 可变；batch 内用 padding+mask。
- **[CELL] token**：晶胞 `L` 编码成一个全局 token，拼在序列最前。
- **晶胞扩散**：默认对 **Gram 6D** 做连续扩散；可选 **Cholesky-6D**（`cell_rep=cholesky6`）以天然保证 SPD。
- **坐标扩散**：对 **分数坐标**做连续扩散，网络用 **v-pred**（推荐稳定）。
- **元素（离散）**：用 **mask diffusion**（简洁、可训练；先跑通再替换为 categorical diffusion 也容易）。
- **物理约束注入**：
  - 训练/采样每步由当前 `(F_t, L_t)` 计算 PBC 最小像距离与 kNN 邻居
  - Transformer 使用 **kNN sparse attention**（复杂度 `O(N·k)`）
  - 同时加 **距离 RBF + 元素对** 的 attention bias（soft constraint）
- **padding 处理**：attention mask + token gating（防残差污染）

---

## 1) 数据预处理（从 CIF/结构到训练张量）

### 1.1 输入字段（单样本）

对每个晶体结构准备：

- `Z`: 原子种类索引，形状 `[N]`，取值 `1..E`（`0` 预留给 PAD）
- `F`: 分数坐标，形状 `[N,3]`，范围建议 wrap 到 `[0,1)`
- `L`: 晶胞矩阵，形状 `[3,3]`

> 建议：先统一选择一种晶胞约定（行/列作为基矢），并在全流程保持一致。

### 1.2 晶胞 6D 表示（Gram6 默认 / Cholesky6 可选）

为减少基矢旋转/等价表述带来的学习困难，使用：

- `G = L^T L`（`3×3` 对称正定）
- 取上三角 6 维向量：
  - `g = [G11, G22, G33, G12, G13, G23]`，形状 `[6]`

训练中对 `g` 做扩散；采样后从 `g` 还原一个合法 `L`（见 §7）。
可选 Cholesky-6D：`y = [log r11, log r22, log r33, r12, r13, r23]`，在 `y` 空间做扩散与 v-pred。

### 1.3 Batch 对齐（padding + mask）

给定 batch 内 `Nmax = max(N_i)`：

- `Z_pad`: `[B, Nmax]`（PAD 填 0）
- `F_pad`: `[B, Nmax, 3]`（PAD 行填 0）
- `mask`: `[B, Nmax]`（真实=1，PAD=0）
- `g`: `[B, 6]`（不 pad）

> 性能建议：用 bucketing（按 N 分桶组 batch）减少 padding 浪费。

---

## 2) 扩散定义（物理叙事：加热熔融 → 熵增）

你可以把 forward diffusion 理解为“热浴扰动导致无序化”。实现上采用标准扩散形式即可。

### 2.1 连续扩散：分数坐标 `F`

- forward：`F_t = α_t F_0 + σ_t ε`（或 EDM 形式用 `σ` 直接控制）
- 训练目标：**v-pred**（推荐，比 ε-pred 更稳）

> 分数坐标注意 wrap：
> - 训练时：对 `F_t` 可以先不 wrap（保持连续），在计算最小像距离时再做 `df - round(df)` 即可
> - 输出/采样时：最终将 `F` wrap 到 `[0,1)`

### 2.2 离散扩散：元素 `Z`（mask diffusion）

- 采样一个 mask 比例 `p_mask(t)`（可随 t 变大而增大）
- 将被 mask 的位置替换为特殊 token `Z=Z_MASK`
- 网络输出 `logits_Z` 预测原始元素

> `Z_MASK` 可映射到 embedding 表的一个专用 id。

### 2.3 连续扩散：晶胞 `g ∈ R^6`（或 Cholesky-6D `y`）

- 对 `g` 做与坐标类似的连续扩散：`g_t = α_t g_0 + σ_t ε`
- 网络同时预测 `v_g` 或 `x0_g`（推荐 v-pred 一致）
> 若启用 Cholesky-6D，扩散变量为 `y`，预测 `v_y` 并在采样时解码回 `g`。

---

## 3) 动态邻居与 PBC 最小像距离（邻接不存储，运行时推导）

### 3.1 最小像距离（必做）

给定分数坐标 `F` 与晶胞 `L`：

1. 分数差：`df = F_j - F_i`
2. 最小像：`df_mic = df - round(df)`（将差折回 `[-0.5, 0.5)`）
3. 笛卡尔位移：`dr = df_mic @ L`（注意你的 L 约定）
4. 距离：`d = ||dr||2`

这一步在扩散每个 step 都基于当前 `F_t, L_t` 计算。

### 3.2 kNN 邻居（推荐）

- 对每个原子 i，取距离最小的 k 个邻居（排除自己与 PAD）
- `k` 推荐：**24 或 32**（第一版用 32 更稳）

输出邻居索引：
- `nbr_idx`: `[B, Nmax, k]`（PAD 行可填 0 并用 mask 屏蔽）
- `nbr_dist`: `[B, Nmax, k]`

> 采样阶段为了省计算：可以每 `s` 步更新一次邻居（推荐 `s=2~5`），但第一版建议每步更新保证正确性。

---

## 4) Token 构造（[CELL] + 原子 token）

### 4.1 原子 token

对每个原子 i：

- 元素嵌入：`eZ_i = Emb(Z_i) ∈ R^{De}`
- 坐标特征：`eF_i = Torus(F_i) ∈ R^{Df}`
  - 对 3D 坐标做多频 `sin/cos` torus 编码（默认 1/2/4/8 频）
- 拼接 + 投影：
  - `x_i = Linear([eZ_i || eF_i]) -> R^D`

batch 形状：
- `X_atom`: `[B, Nmax, D]`

### 4.2 [CELL] token（方案 A）

- `x_cell = MLP_cell(g_t) -> R^D`
- `X_cell`: `[B, 1, D]`

### 4.3 拼接成序列

- `X = concat([X_cell, X_atom], dim=1)`
- `X`: `[B, 1+Nmax, D]`

序列 mask：
- `seq_mask`: `[B, 1+Nmax]`
  - `seq_mask[:,0]=1`（CELL 永远有效）
  - `seq_mask[:,1:]=mask`

---

## 5) Transformer 架构微调（DiT-style + 稀疏邻居注意力）

### 5.1 条件注入（时间步/温度）

- `t` 或 `σ` -> `SinCos` -> `MLP_t` 得到 `c_t ∈ R^D`
- 在每个 block 用 **AdaLN（FiLM）** 调制 LN：
  - 生成 `(γ1,β1,γ2,β2)` ∈ `R^{B×D}`

> 直觉：`t` 越大（噪声越大）对应“温度更高/熵更高”，AdaLN 让网络显式感知该状态。

### 5.2 注意力：kNN sparse attention + bias（核心）

你将实现一种“邻居引导的 Transformer 注意力”，但主体仍是 Transformer。

#### Query/Key/Value
- 标准多头：`Q,K,V: [B, heads, Nseq, dh]`
- 其中 `Nseq = 1+Nmax`

#### 稀疏化策略
- **仅对原子 token 做 kNN 稀疏注意力**：
  - 原子 i 只 attend 到其 `k` 个邻居 token
- 对 `[CELL]` token：
  - 建议让 CELL attend 到所有原子（全局汇聚）
  - 让所有原子也可 attend 到 CELL（全局条件传播）

实现时最简单的做法：
- 原子-原子：稀疏（index-gather 的方式计算注意力）
- CELL 相关：保留全连接（1×N 与 N×1 很便宜）

### 5.3 Attention bias（软物理约束）

对每条边 `(i -> j)`（i 的邻居 j）：

- 距离 RBF：`rbf(d_ij) -> R^K`（K=32/64）
- 元素对 embedding（可选但推荐）：`pair(Z_i,Z_j) -> R^{Dp}`
- `bias_ij = MLP_bias([rbf || pair]) -> R^{heads}`

把它加到注意力 logits：
- `logits[b,h,i,neighbor_slot] += bias_ij[b,h]`

> 这比硬编码键合规则更稳：它不会禁止边，只会偏置注意力。

### 5.4 padding 的正确处理（必须同时做两件事）

1) **Attention mask**：
- PAD token 不可作为 key/value
- 邻居列表中若出现 PAD，直接在对应 logits 加 `-inf`

2) **Token gating**（强烈推荐）：
- 在每个 block 末尾或至少在输出头前：
  - `X = X * seq_mask[...,None]`
- 防止 PAD token 通过残差/MLP 漂移污染

---

## 6) 输出头与训练损失

### 6.1 输出头

从 Transformer 最终输出 `X_out: [B, 1+Nmax, D]`：

- `[CELL]` token 输出用于晶胞：
  - `pred_v_g = Head_g(X_out[:,0,:]) -> [B,6]`

- 原子 token 输出用于：
  - 坐标：`pred_v_F = Head_F(X_out[:,1:,:]) -> [B,Nmax,3]`
  - 元素：`logits_Z = Head_Z(X_out[:,1:,:]) -> [B,Nmax,E]`

> 说明：你也可以把晶胞 head 做成独立小网络；但用 CELL token head 更统一。

### 6.2 损失函数（mask 加权）

- 坐标（连续）：
  - `L_F = MSE(pred_v_F, v_target_F)`，仅对 `mask=1` 的原子求和/平均
- 元素（离散 mask 预测）：
  - `L_Z = CE(logits_Z, Z_0)`，仅对被 mask 的位置计算（或对所有真实原子计算）
- 晶胞（连续）：
  - `L_g = MSE(pred_v_g, v_target_g)`

总损失：
- `L = L_F + λ_Z L_Z + λ_g L_g`

经验初值：
- `λ_Z = 1.0`
- `λ_g = 0.1 ~ 1.0`（看 g 的尺度，必要时对 g 做标准化）

---

## 7) 采样流程（生成晶体）

### 7.1 初始化

- `F_T ~ Normal(0,1)`（形状 `[B,Nmax,3]`，PAD 行置 0）
- `Z_T = Z_MASK`（所有真实原子位置都 mask；或采样一个起始分布）
- `g_T ~ Normal(0,1)`（形状 `[B,6]`）；若 Cholesky-6D 可选 `y_iso=[log s0, log s0, log s0, 0,0,0] + N(0,σ)`

> 若你还未解决“生成原子数 N”的问题：先固定 N（例如从数据分布采样 N 或条件给定 N）。

### 7.2 反向扩散迭代

对 `t = T..0`：

1) 由 `g_t` 还原 `L_t`（见下一节）
2) 用 `F_t, L_t` 计算 PBC 距离与 kNN 邻居
3) 构造 token：`[CELL](g_t) + atoms(Z_t,F_t)`
4) 前向网络得到 `pred_v_F, logits_Z, pred_v_g`
5) 用采样器更新：
   - 坐标：`F_{t-1} = sampler_step(F_t, pred_v_F, t)`
   - 晶胞：`g_{t-1} = sampler_step(g_t, pred_v_g, t)`
   - 元素：根据 `logits_Z` 逐步反 mask（例如从高噪声到低噪声逐渐减少 mask 数量）

最后：
- `F_0` wrap 到 `[0,1)`
- `Z_0` 取 argmax 或采样
- `L_0` 由 `g_0` 还原

### 7.3 从 g(6D) 还原一个合法 L(3×3)

因为 `G = L^T L` 正定，你可以：

1) 从 `g` 组装对称矩阵 `G`
2) 对 `G` 做 Cholesky：`G = R^T R`
3) 取 `L = R`（或 `L = R^T`，只要全流程一致）

必要的数值保护：
- 保证 `G` 正定：
  - 对角加 `eps`（如 `1e-6`）
  - 若出现非正定，做投影（最近正定矩阵）或丢弃该样本

---

## 8) 关键工程细节（决定你能否训稳）

### 8.1 坐标尺度与数值稳定

- 分数坐标在 `[0,1)`，扩散噪声要与该尺度匹配
- v-pred 通常更稳；若不稳，可尝试 x0-pred

### 8.2 邻居构建在高噪声阶段的鲁棒性

- 扩散早期 `F_t` 噪声大，邻居可能不完全正确
- 你已用 **bias** 软约束，能缓解邻居误差
- k 建议先大（32），训稳后再减（24/16）做消融

### 8.3 CELL token 的注意力连接

- 强烈建议：
  - 原子 token 都可 attend 到 CELL
  - CELL 可 attend 到所有原子
- 这让晶胞信息在网络里传播最顺畅

### 8.4 PAD 的双重屏蔽（必须）

- attention mask 屏蔽 PAD 作为 key/value
- token gating 清零 PAD token（每层或输出前）

### 8.5 训练/采样速度

- kNN sparse attention 是省算力关键
- 邻居可每 2~5 步更新一次（采样时）

---

## 9) 最小可用 checklist（照着实现就能跑）

### 数据侧
- [ ] 从结构解析出 `(Z, F, L)`
- [ ] `F` wrap `[0,1)`
- [ ] `g = upper_tri(L^T L)`
- [ ] batch padding 得到 `Z_pad, F_pad, mask, g`

### 扩散侧
- [ ] 定义连续扩散（坐标 F 与晶胞 g）
- [ ] 元素 mask diffusion（含 Z_MASK id）
- [ ] 训练 target 用 v-pred（F 与 g）

### 模型侧
- [ ] 构造原子 token：`Emb(Z)+Fourier(F)->Linear->D`
- [ ] 构造 CELL token：`MLP(g)->D`
- [ ] 拼接序列：`[CELL]+atoms`
- [ ] AdaLN 条件：`t->c_t` 调制每层 LN
- [ ] 动态邻居：MIC 距离 + kNN
- [ ] sparse attention：原子-原子按 kNN；CELL 全连接
- [ ] attention bias：RBF(dist)+pair(Zi,Zj)->head bias
- [ ] PAD：attention mask + token gating

### 输出侧
- [ ] `Head_F` 输出 `pred_v_F` (B,N,3)
- [ ] `Head_Z` 输出 `logits_Z` (B,N,E)
- [ ] `Head_g` 从 CELL 输出 `pred_v_g` (B,6)

### 损失
- [ ] `L_F` mask 加权
- [ ] `L_Z` 对真实/被mask位置计算
- [ ] `L_g` 连续 MSE

### 采样
- [ ] 逐步更新 `F_t, g_t, Z_t`
- [ ] `g_t -> G -> chol -> L_t`
- [ ] 最终 wrap `F_0`，解码 `Z_0`

---

## 10) 关键缺口的细化方案（收敛、可行性、实现一致性、性能）

本节针对你指出的 6 个缺口给出可直接落地的规格，避免实现分歧。

---

### 10.1 离散/连续耦合的联合目标、权重与采样顺序

#### 10.1.1 统一时间步与噪声调度（建议）
- 使用同一个连续噪声标量 `σ`（或离散 `t`）驱动三路扰动：坐标 `F`、晶胞 `g`、元素 `Z`。
- 让三路的“难度”随 σ 同步上升，避免某一路提前收敛、另一条一直学不动。

建议映射：
- 连续：`F_σ = F + σ·ε_F`，`g_σ = g + σ·ε_g`
- 离散 mask 概率：`p_mask(σ) = clamp(p_min + (p_max-p_min)·s(σ), 0, 1)`
  - `s(σ)` 可取 `σ/σ_max` 或 log-space 归一化
  - 推荐：`p_min=0.05, p_max=0.60`（元素不要全程都 mask 太高，否则条件不足）

#### 10.1.2 预测参数化（强推荐 v-pred）
- 坐标 head：预测 `v_F`
- 晶胞 head：预测 `v_g`
- 元素 head：预测 `Z0`（分类 logits）

> 若你使用 EDM/连续 σ，v-pred 的公式按你选的噪声参数化保持一致即可。关键是：F 与 g 采用同一种连续目标，便于平衡。

#### 10.1.3 损失设计（联合训练，动态平衡）
定义：
- `L_F = mean_mask( ||v̂_F - v_F||^2 )`（只对真实原子）
- `L_g = ||v̂_g - v_g||^2`（每样本 6D）
- `L_Z = CE(logits_Z, Z0)`（只对被 mask 的位置；也可对全原子但加权）

**推荐的稳定权重策略（无需手调）**：不确定性加权（homoscedastic uncertainty）
- 引入可学习标量 `s_F, s_g, s_Z`
- 总损失：
  - `L = exp(-s_F) L_F + s_F + exp(-s_g) L_g + s_g + exp(-s_Z) L_Z + s_Z`

好处：三路梯度会自动平衡，收敛更稳。

**若你不想引入可学习权重**，给一个可用默认：
- 先对 g 做标准化（z-score）再训练
- `λ_Z=1.0`
- `λ_g=0.5`
- 坐标为主：`L = L_F + λ_Z L_Z + λ_g L_g`

#### 10.1.4 采样顺序（并行、同一步更新，推荐）
为避免“先定 L 再定 F”带来误差累积，推荐每一步并行更新：
- 输入当前 `(F_σ, g_σ, Z_masked)` -> 同一次前向输出 `(v̂_F, v̂_g, logits_Z)`
- 在同一 σ-step 内同时更新：
  - `F <- step(F, v̂_F, σ)`
  - `g <- step(g, v̂_g, σ)`
  - `Z <- unmask_step(Z, logits_Z, σ)`

元素 unmask 策略（稳）：
- 设定每个 σ 的目标 mask 比例 `p_mask(σ)`，逐步减少
- 每步选择一部分 mask 位点：
  - 取 `max_prob = max softmax(logits_Z)` 最大的若干位点先填（最确定的先确定）
  - 其余保持 mask 到更低 σ 再填

> 若你担心并行导致不稳定，可在同一 σ-step 内做 **两次交替细化**（但仍共用 σ）：
> 1) 用当前 Z 更新 F,g；2) 用更新后的 F,g 再 refine logits_Z（额外一次前向）。第一版不建议，先并行跑通。

---

### 10.2 晶胞 6D 采样与恢复（SPD、尺度裁剪、规约）

#### 10.2.1 g -> G 的组装
`g = [G11, G22, G33, G12, G13, G23]`
- 组装对称矩阵：
  - `G = [[G11,G12,G13],[G12,G22,G23],[G13,G23,G33]]`

#### 10.2.2 正定性保证（推荐：Cholesky with jitter + fallback）
**主流程（快）**：
1) `G' = symmetrize(G)`（数值对称）
2) jitter：`G'' = G' + ε·I`，`ε` 从 `1e-6` 起，失败则倍增到 `1e-2`
3) 尝试 Cholesky：`G'' = R^T R`
4) 取 `L = R`（或 `R^T`，全流程一致）

**fallback（稳）**：若 Cholesky 多次失败
- 做 eigen 分解：`G = Q diag(λ) Q^T`
- 裁剪：`λ_clamp = clamp(λ, λ_min, λ_max)`
  - 推荐：`λ_min=1e-4`（避免塌缩），`λ_max` 取数据集 99.5% 分位对应尺度
- 重建：`G_spd = Q diag(λ_clamp) Q^T`
- 再 Cholesky 得 `L`

#### 10.2.3 尺度与形状裁剪（防极扁晶胞）
为了避免“体积极小/极大、角度奇异”的晶胞：

- 体积：`V = det(L)`
  - 在每步或最终：将 `V` 限制到数据集分位区间 `[V_p1, V_p99]`
  - 实现：若 `V` 超界，对 `L` 做各向同性缩放 `L <- L * (V_target/V)^{1/3}`

- 条件数：`cond(G)` 或 `max(λ)/min(λ)`
  - 限制到 `cond_max`（如 1e3 或按数据分布）

#### 10.2.4 晶胞规约（可选但推荐：Niggli 或近似规约）
- 训练时：建议对所有样本做一次统一规约（Niggli/LLL 等），减少等价晶胞多样性。
- 采样后：对生成的 `L` 再做规约，以获得标准表示并降低后处理失败率。

> 若你不想引入严格规约算法，至少做一个“排序/正交化近似”即可：
> - 将三基矢按长度排序（短到长）
> - 强制右手系（det>0）

---

### 10.3 邻居与 CELL token 的联动（维度、是否包含 CELL、bias 规则）

#### 10.3.1 序列定义与索引
- 序列 token：`0` 为 CELL，`1..Nmax` 为原子 token
- `seq_mask[:,0]=1`，原子用 `mask`

#### 10.3.2 邻居列表是否包含 CELL？（推荐：不包含）
- kNN 邻居仅在 **原子集合** 上构建：原子 i 的邻居 j ∈ {原子}。
- CELL 是全局 token，不属于几何邻居体系；**不要放进 kNN 列表**。

#### 10.3.3 注意力连接规则（推荐、实现明确）
- Atom -> Atom：稀疏（kNN）
- Atom -> CELL：允许（全连接 1 条边/原子，额外开销 O(N)）
- CELL -> Atom：允许全连接（O(N)）
- CELL -> CELL：自连接

对应实现（logits 结构）：
- 原子-原子 logits 用稀疏张量 `[B, heads, N, k]`
- 与 CELL 相关两项：
  - `logits_atom_to_cell`: `[B, heads, N, 1]`
  - `logits_cell_to_atom`: `[B, heads, 1, N]`
  - 拼接/合并时保持一致即可（或单独算再合成输出）

#### 10.3.4 CELL 相关 bias/距离怎么处理？
- 不对 CELL 定义几何距离（避免引入不必要假设）
- CELL 相关 bias 采用 **可学习常量/线性**：
  - `bias_atom_to_cell = MLP_ac([x_atom || x_cell || c_t]) -> heads`
  - `bias_cell_to_atom = MLP_ca([x_cell || x_atom || c_t]) -> heads`

> 简化版：CELL 相关不加 bias（设为 0）也可先跑通。

---

### 10.4 padding 安全性（kNN 构建前必须过滤 PAD，避免 NaN/崩溃）

#### 10.4.1 距离计算前的 PAD 过滤
- 仅对真实原子计算距离：使用 `mask` 生成有效索引列表 `idx_valid`。
- 对 PAD 行/列不要参与 pairwise 距离计算。

推荐实现方式（安全且简单）：
- 先将 PAD 原子坐标置为一个极端值并配合 mask（不推荐，易溢出）
- **推荐**：直接在距离矩阵上做 masked fill：
  - 计算全 `Nmax×Nmax` 的 df_mic（向量化）
  - 对任一端是 PAD 的 pair：`dist = +inf`
  - 对 i=j：`dist = +inf`
  - 然后 kNN 在 `+inf` 上不会选到 PAD

这样即使后续 logits 屏蔽，前面也不会因为无效值导致 NaN。

#### 10.4.2 kNN 输出的 mask
- `nbr_idx` 若因 N<k 出现无效位置：用 0 填充并产生 `nbr_mask`：`[B,N,k]`
  - `nbr_mask=1` 表示有效邻居
  - attention logits 对 `nbr_mask=0` 加 `-inf`

---

### 10.5 物理约束补强（后验合法性检查 + 质量评估指标）

扩散与 bias 是“软引导”，但你需要 **后验可行性检查** 来确保生成物理合理，并为训练/消融提供可量化指标。

#### 10.5.1 采样后必做的合法性检查（建议作为 filter）
1) **最小间距（hard）**
- 用 PBC 最小像距离计算所有对的 `d_ij`
- 若 `min(d_ij) < d_min` 则判为碰撞
- `d_min` 可取元素相关半径：
  - 简化：统一阈值 0.7–1.0 Å（依数据）
  - 更好：`d_min(i,j)=s*(r_cov(Zi)+r_cov(Zj))`，`s≈0.7~0.9`

2) **晶胞体积范围**
- `V in [V_p1, V_p99]`（按训练集分位）

3) **成分/元素分布约束（可选）**
- 若你有化学式条件：检查生成元素计数是否匹配
- 若无条件：至少限制“罕见元素概率”或按训练集先验重加权（防跑飞）

4) **电中性/化学合理性（可选）**
- 若你有价态/氧化态先验，可做快速电中性检查（第一版可不做）

#### 10.5.2 质量评估与可视化（训练/推理都用得上）
建议记录以下指标（每个 batch 统计均值/分布）：
- `valid_rate`: 通过上述合法性检查的比例
- `min_dist` 分布、`V` 分布、`cond(G)` 分布
- `coord_MAE` / `RMSD`（对齐到真值时，训练/验证用）
- `element_acc`（mask 位点 top-1）

如果你有 XRD 模拟器/代理模型：
- 增加 `XRD_consistency`（生成结构的 XRD 与目标/先验的一致性）作为额外评估/奖励

---

### 10.6 训练/推理性能与显存建议（可直接按此落地）

#### 10.6.1 推荐模型规模区间（从小到大）
以 Nmax≈24–128 为常见范围给出可用配置（你可从最小开始）：

- **Small（单卡友好）**
  - `D=256`, `heads=8`, `layers=8`, `mlp_ratio=4`
  - `k=24` 或 `32`

- **Base（性能更强）**
  - `D=384`, `heads=12`, `layers=12`, `mlp_ratio=4`
  - `k=32`

- **Large（更吃显存）**
  - `D=512`, `heads=16`, `layers=16`, `mlp_ratio=4`
  - `k=32`

经验：先用 Small 跑通并观察 `valid_rate`，再放大。

#### 10.6.2 混合精度与数值稳定
- 训练用 `bfloat16`（优先）或 `fp16`（更易溢出）
- 距离计算/Cholesky/eig 建议用 `fp32`（关键数值步骤保持稳定）

#### 10.6.3 降显存策略
- 梯度 checkpoint：对每 2–4 个 block checkpoint 一次
- 关闭不必要的中间保存（attention weights 不回传存储）
- 使用稀疏注意力后，显存主要来自 token 激活与 MLP

#### 10.6.4 推理加速策略
- 采样步数：优先用 20–30 步（DDIM/EDM sampler），再做质量-速度权衡
- 邻居更新：每 2–5 步更新一次 kNN（采样时）
- 元素解 mask：低 σ 末段集中解码（减少早期波动）

---

## 11) 你接下来需要补齐的最小信息（用于与你现有 SCDM 管线对接）

为把你现有 `B×3×24×24` 数据无缝迁移到本路线，只需明确两件事：

1) 三个通道分别存了什么（元素/坐标/晶格/其他）
2) 24×24 的两轴语义（slot vs feature 或 token-token 关系）

一旦你给出这两点，我可以把 **SCDM -> (Z,F,L,mask)** 的解包规则写成逐字段公式，确保你不用推倒重来。

---

## 12) 缓存与邻居重用（落地方案）

适用于当前“动态邻居 + 稀疏注意力”实现，目标是降低采样/训练中 MIC+kNN 的重复开销。

- **缓存键**：按 batch 分桶，键由 `(Nmax, k, cell_hash, step_mod)` 构成，其中 `cell_hash` 取 `g` 量化（例如把 `g`/`σ_g` 取整到 1e-3），`step_mod` 用于每 `s` 步强制刷新。
- **缓存值**：`nbr_idx`, `nbr_dist`, 以及对 bias 用的 `rbf(dist)`。为防占显存，缓存存 CPU（`pin_memory=True`），用时拷回 GPU。
- **失效/刷新**：若 `||g_t-g_prev||` 或 `||F_t-F_prev||`（均方）超阈值则丢弃缓存；否则沿用并按 `step_mod` 触发重算。失败 fallback：直接重算当前 batch。
- **PAD 安全**：kNN 之前用 `mask` 把 PAD 对应的距离填 `+inf`，并产出 `nbr_mask`；缓存时连同有效原子数存储，回放时只对前 `n_atoms` 位置使用，避免 PAD 污染 logits。
- **并行复用**：采样时同一 bucket 内多个样本可共用缓存键；CELL 不参与 kNN，省去额外维度。
- **监控**：记录缓存命中率、重算耗时，作为调节 `s`、量化精度和阈值的依据。

---

## 13) Flow-Matching 训练/采样对齐（替换扩散定义）

将 §2/§7 的扩散与采样统一为 Flow-Matching/概率流 ODE，避免训练/采样分布漂移。

- **时间采样**：训练采样 `t ~ U(0,1)`，采样用同一 `t` 网格（线性或余弦）。
- **连续变量 (F, g)**：
  - 构造 `x_t = t·x_1 + (1-t)·x_0`，`x_1` 取高斯先验（训练/采样一致）。
  - 目标速度 `v = x_1 - x_0`，损失 `||v̂ - v||^2`，可加权 `w(t)`（如 `(1-t)`）。
  - 采样用 Euler/Heun 对 `dx/dt = v̂(x,t)` 积分，步长与邻居刷新间隔对齐（例如 20–30 步，kNN 每 2–3 步）。
- **离散元素 Z（mask diffusion 保留）**：
  - `p_mask(t)` 随 t 单调递增（线性/余弦），训练在同一 t 上计算 CE。
  - 采样同一 t 网格上逐步解 mask：按 `p_mask(t)` 的目标比例，优先解码置信度最高的若干位点。
- **三路同步**：`F/g/Z` 共用同一 t；同一步前向输出 `(v̂_F, v̂_g, logits_Z)`，并行更新三路以减少相互漂移。
- **晶胞稳定性**：`g` 采样后 SPD 投影（Cholesky + jitter），并对体积/条件数做裁剪，防止数值爆炸。

### 方案落地适配性评估
- 与当前 3×24×24 网格实现相比，Flow-Matching 省去了 logit-normal 调度，训练/采样调度一致性更好；需要为 `F/g` 引入 `x_1` 先验（高斯）且统一 t 网格。
- 缓存方案与动态邻居设计兼容，代码侵入点清晰（kNN 构建前/后）；但若未来 N 很大或 k 降低，仍需监控命中率与显存（CPU 缓存占用）。
- 核心风险仍在晶胞 SPD 投影和最小间距过滤：需在采样后增加合法性检查，以保证生成质量。

---

## 14) 优化与风险收敛计划（按可落地优先级）

### 14.1 kNN + MIC 复杂度瓶颈
- **两阶段候选**：先在分数坐标 [0,1)^3 做 hash/grid 分箱（G≈8–20），仅对自身格子和 26 邻格形成候选对；再对候选对做 MIC（`df_mic = df - round(df)`，`d² = df_micᵀ G df_mic`）并 top-k。这样把 N² 降到 O(N·k_cand)。
- **Verlet/skin 重建触发**：缓存邻居时同时存上一帧坐标，只有当最大位移 > skin/2 时才重建候选（flow ODE 步进下位移可预估）；可与 §12 缓存键结合。
- **分块 MIC**：用度量矩阵公式避免 27 images，配合分块 bmm/einsum + running top-k，减少峰值显存。

### 14.2 Flow-Matching 的离散/先验细化
- **p_mask(t), w_Z(t) 三档可消融**：线性 / 余弦 / sigmoid 三个 schedule，元素解码时机与几何稳定期对齐；w_Z(t) 同步调节 CE 权重，建议用有效性指标做对比。
- **混合先验**：连续变量一部分 batch 用高斯采样的 `x_1`，一部分用“数据扰动 x_0+ε”；`g` 和 `F` 共享 t 网格。元素 Z 可用数据集先验（或条件化先验）初始化，降低跑飞概率。

### 14.3 晶胞合法性前置
- **参数化保证 SPD**：网络输出下三角 A，对角 softplus+eps，设 `G = A Aᵀ + eps·I`，天然 SPD，基本不再需要昂贵的“最近 SPD 投影/丢弃”。
- **体积/条件数正则**：对 log(V) 和 cond(G) 设平滑 penalty（或 sigmoid 压缩到分位区间），采样后仍做体积/角度范围 filter。

### 14.4 PAD 机制 + 断言 + 单测
- **机制**：距离矩阵对 PAD/+self 填 +inf，kNN 产生 `nbr_mask` 并在 logits 上 -inf，Transformer 保持 attention mask + token gating。
- **断言**：forward 时检查 finiteness、`nbr_idx < n_atoms`、PAD 输出为 0；debug/单测默认开启。
- **单测**：小 batch（多种 n_atoms）跑邻居构建+注意力，暴力对比 nbr_mask 与 brute-force 结果，确保无 NaN。

### 14.5 采样质量评测与闭环
- **后验 filter**：PBC 最小间距（元素半径/统一阈值）、晶胞体积范围、元素分布/化学式一致性（有条件则硬匹配）。
- **指标面板**：valid_rate、min_dist 分布、V/cond(G) 分布、element_acc（mask 位点 top-1）；可选加 StructureMatcher 匹配率/RMSD 或 PXRD agreement 作为领域指标。
- 这些指标用来比较不同 p_mask/w_Z 方案、Verlet 与否、SPD 参数化与否，形成迭代闭环。

---

## 15) 与当前 2DGEN 实现对齐的落地修正（必须先做）

> 本节是对现有 `2DGEN/` 代码结构的 **适配说明**，用于把本指南从“理念”变成可跑的落地版本。

### 15.1 数据与表示对齐（从网格切换到 token）
- 现有实现是 **3×24×W 网格**；要落地本方案需要 **Z/F/L 的 token 表示**。
- 落地改法：基于 `C2DBDataset` 直接输出 `(Z, F, mask, L)`，训练时即时计算 `g=upper_tri(L^T L)`。
- 为稳定训练，对 `g` 做固定缩放：`g_scaled = g / g_scale`（例如 `g_scale = lattice_scale^2` 或按数据统计给定）。

### 15.2 扩散调度与 v-pred 对齐（先与当前实现一致）
- 现有 `common/diffusion.py` 使用 `t ~ sigmoid(N(P_mean,P_std))`，并定义
  - `z = t·x0 + (1-t)·noise`，`v = (x0 - z)/(1-t)`
- 为减少改动，**F 与 g 先沿用同一套调度**，保证训练与采样一致。
- 元素 mask 概率 `p_mask(t)` 用线性或余弦从 `p_max -> p_min`，以 t 为自变量。

### 15.3 kNN 注意力的第一版实现（避免自定义稀疏算子）
- Nmax<=24 时全连接开销很小；但仍需模拟“kNN 约束”：
  - 先算 `N×N` 距离矩阵，再取 top-k 形成 **attention mask**；
  - 实际计算仍是 dense attention，但 logits 对非邻居设 `-inf`。
- 这样能复用标准注意力代码，同时与指南的“邻居稀疏化”保持一致。

### 15.4 CELL 与 Z_MASK 处理
- CELL 不进 kNN 列表；所有 atom 允许 attend CELL，CELL 允许 attend 所有 atom。
- Z 被 mask 时仍参与 pair bias：使用 `Z_MASK` 的 embedding 参与计算即可。

### 15.5 N 的生成策略（避免采样阶段卡住）
- 训练：沿用 `max_atoms` + padding mask。
- 采样：**从训练集 N 分布采样 N**（或显式指定 N），再按该 N 设 mask。

### 15.6 SPD/数值稳定优先级
- 训练：每次用 `g_t` 恢复 `L_t` 前先做 SPD jitter；失败再做 eig clamp。
- 采样：最终一定要做 SPD 投影 + 体积/条件数裁剪。

### 15.7 代码落地路径（建议）
- 新增 token 版模型与扩散模块，先不移除原网格实现，避免回归。
- 提供新的 `train_tokens.py` 与 `sample_tokens.py`，并在 `README` 中并列说明。

---

## 16) 当前落地状态（已实现 / 未实现）

> 用于跟踪 `2DGEN/` 代码是否对齐本指南。标注对应落地点或缺口位置。

### 16.1 已实现（核心路线）
- **token 表示 + CELL**：`model/atom_transformer.py`
- **Gram6/Cholesky6 晶胞扩散 + v-pred**：`common/atom_diffusion.py`, `model/atom_denoiser.py`
- **动态邻居 + MIC + kNN gather attention**：`common/crystal.py`, `model/atom_transformer.py`
- **RBF + 元素对 bias**：`model/atom_transformer.py`
- **PAD 双重屏蔽**（attention mask + token gating）：`model/atom_transformer.py`
- **N 分布采样**（基于 npz 统计）：`sample_tokens.py`
- **采样合法性过滤**（体积范围 + 最小间距）与 `valid_rate`：`sample_tokens.py`
- **动态损失权重**（uncertainty weighting）：`common/atom_diffusion.py`
- **Flow-Matching 可选**：`common/atom_diffusion.py`, `train_tokens.py`
- **npz token 缓存训练**：`data/c2db_dataset.py`, `train_tokens.py`

### 16.2 未完全实现 / 可选优化
- **邻居隔步更新**（s 步刷新）：仅做了缓存阈值复用，未实现 step 周期刷新
- **晶胞规约**（Niggli/LLL）：当前仅做“排序+右手系”的简化规约
- **评估指标面板**（min_dist/cond(G)/分布统计）：未系统化记录到训练/采样日志

---

### 16.3 当前 token 线实现逻辑（训练/采样/评估）

> 目的：明确当前代码已经落地的行为，避免评估与调参时误解。

#### 16.3.1 数据与表示（npz）
- 数据缓存（如 `data/C2DB/ache/c2db_tokens.npz`）包含：`z`、`f`、`atom_mask`、`lattice`、`gram6`、`g_scale`。
- `z` 为原子序数（含 mask/pad），`f` 为分数坐标，`gram6` 为 Gram 6D 表示（训练可直接用）。

#### 16.3.2 训练入口与损失（token diffusion/flow）
- 入口脚本：`scrip/train_tokens.py`（默认走 token 线）。
- 数据来源：优先用 `C2DBTokenNPZDataset(npz)`，否则从 CSV 在线构建 `C2DBAtomDataset`。
- 损失函数：`AtomVelocityLoss`  
  - 连续部分：`v_f`（frac）与 `v_g`（gram6）的 MSE  
  - 离散部分：`z` 的 masked CE  
  - 默认开启 `use_uncertainty_weighting`（`s_f/s_g/s_z` 学习缩放）
- 时间采样：diffusion 模式用 logit-normal；flow 模式用 U(0,1)。
- `p_mask(t)` 控制 `z` 的 mask 比例（flow 递增、diffusion 递减）。
- 训练权重保存：每次训练在 `outputs/checkpoints/<YYYYMMDD_HHMMSS>/` 下写 `atomdenoiser_last.pt`、`atomdenoiser_best.pt` 与 `config.json`。

#### 16.3.3 采样逻辑（生成）
- 入口脚本：`scrip/sample_tokens.py`
- 初始化：`z` 全 mask，`frac/gram6` 高斯噪声；Cholesky-6D 可用 `cell_init=iso` 并自动估计尺度。
- 采样：按 `steps` 逐步更新 `frac/gram6`，同时按 `p_mask` 解码 `z`。
- `z` 解码策略可选：`argmax/temperature/topk/topp`（默认 `temperature`）。
- 晶胞恢复：`gram6_to_lattice(g * g_scale)` + `clip_lattice(v_min/v_max/cond_max)`，可选 `reduce_lattice`/`niggli_reduce`。

#### 16.3.4 采样后过滤与输出
- 默认过滤：体积范围（来自 `--npz` 的分位数）+ `min_dist`。
- 输出：`samples.npz` + CIF（只对通过过滤的样本）。

#### 16.3.5 评估入口（已落地的部分）
- `scrip/eval_samples.py`：输出 Tier-0/1 指标（json + per-sample）
- `scrip/plot_eval.py`：绘制直方图与厚度-真空散点图
- `scrip/plot_compare.py`：样本 vs 训练集分布的半小提琴对比图

### 16.4 模型架构与张量流（当前实现）

#### 16.4.1 输入/输出与张量形状
- 输入张量（训练/采样一致）：
  - `z`: `(B, N)`，原子序数 token（pad=0，mask_id=num_elements+1）
  - `frac`: `(B, N, 3)`，分数坐标
  - `gram6`: `(B, 6)`，Gram 6D 表示
  - `atom_mask`: `(B, N)`，1 表示有效原子
  - `t`: `(B,)`，时间步
- 模型输出：
  - `pred_f`: `(B, N, 3)`，frac 的速度预测 `v_f`
  - `pred_g`: `(B, 6)`，gram6 的速度预测 `v_g`
  - `pred_z`: `(B, N, num_elements+1)`，元素 logits（不含 pad）

#### 16.4.2 Token 构造与嵌入
- `z`：Embedding(`num_elements+2`) -> `z_emb` `(B, N, z_dim)`
- `frac`：Fourier features（`fourier_freqs`）-> `f_emb` `(B, N, f_dim)`
- 原子 token：`[z_emb, f_emb]` concat -> `in_proj` -> `(B, N, embed_dim)`
- CELL token：`gram6` -> `cell_mlp` -> `(B, 1, embed_dim)`
- 拼接序列：`tokens = [CELL] + atom_tokens` -> `(B, N+1, embed_dim)`
- 时间条件：`t` -> sinusoidal embedding -> `time_mlp` -> `cond` `(B, embed_dim)`

#### 16.4.3 邻居构建与注意力
- 只对原子 token 做 kNN（CELL 不参与邻居）。
- `gram6 * g_scale` -> `lattice`，用 MIC 计算 `dist` `(B, N, N)`。
- `dist` -> `nbr_idx/nbr_mask` `(B, N, k)`，并取 `dist_nbr`。
- 距离 -> RBF -> `rbf` `(B, N, k, rbf_dim)`。
- 结合 `z_emb` 形成 pair bias -> `bias_nbr` `(B, heads, N, k)`。
- `AtomBlock` 里用稀疏 gather attention（原子-原子）+ CELL 交互 bias。
- `atom_mask` 在每层做 token gating，清零 PAD 残差污染。

#### 16.4.4 Head 输出与损失对应关系
- `pred_f`：来自 atom token head，匹配 `v_f`（frac 速度）
- `pred_g`：来自 CELL token head，匹配 `v_g`（gram6 速度）
- `pred_z`：来自 atom token head，针对 mask 位点做 CE
- 对应损失：`AtomVelocityLoss = loss_f + loss_g + loss_z`（可选不确定性加权）

#### 16.4.5 采样时的张量流
- 初始化：`z` 全 mask，`frac/gram6` 为高斯噪声。
- 每步预测 `pred_f/pred_g/pred_z`，更新 `frac/gram6`，按 `p_mask` 逐步解码 `z`。
- 采样结束：`gram6` -> `lattice` -> 裁剪体积/条件数 -> 可选规约（Cholesky-6D 先解码回 `gram6`）。

## 17) 评估任务规划（适配当前 2DGEN 任务与代码）

本节用于把评估闭环落到当前 token-based 2DGEN（无条件生成/可选 reconstruction），并明确哪些评估在当前阶段是必做、哪些是可选。

### 17.1 评估范围与边界（当前阶段）
- 当前模型任务：基于 `(Z, F, g)` 的无条件生成（token diffusion/flow）。
- 不在当前阶段：XRD 条件一致性、DFT 能量/声子等高成本物理验证。
- 可选任务：reconstruction/denoising 评估（需要从真值结构加噪再反推）。

### 17.2 评估层级与优先级（按成本/产出比排序）
Tier-0（每次采样必做，秒级～分钟级）
- `valid_rate`（体积范围 + 最小间距 + SPD + 规则过滤）
- 分布统计：`min_dist`、`volume`、`cond(G)`、`N_atoms`
- 元素分布：生成 vs 训练集（直方图或 top-K 频率表）

Tier-0.1（SCDM 风格硬过滤，快速拒绝）
- `N_atoms >= 3`、空样本过滤
- 坐标重复过滤（eps ~ 1e-3）
- `min_dist` 硬阈值（默认 1.5 A，可配置）
- 2D 快速门槛（厚度/真空基于训练集分位数）

Tier-1（二维材料关键指标，分钟级）
- 厚度 `thickness` 与真空 `vacuum`
- `cross_vacuum_bond`（跨真空方向近邻判定）
- `gcc_ratio`（面内连通性）
- `anisotropy`（|c| / mean(|a|,|b|)）

Tier-2（可选：reconstruction 评估，小时级）
- 仅在需要验证结构分布学习能力时做
- 指标优先级：RMSD / Sinkhorn / CrystalNN / XRD / OFM（按依赖与稳定性逐步补齐）
- 加入 `15% displacement` 基线（用于证明模型确实在纠错）

Tier-3（可选：采样集去重与新颖性，小时～天级）
- `unique_rate`：StructureMatcher 去重
- `novel_rate`：与训练集匹配后的新颖比例

### 17.3 评估输入与采样协议（当前任务适配）
- 无条件生成：从先验采样，固定 `seed` 与 `num_samples`（建议 1k）。
- N_atoms 选择：与 `sample_tokens.py` 一致，从训练集 N 分布抽样或显式指定。
- 统计对齐：使用训练集分位数作为过滤与分布对齐基准（V/min_dist/anisotropy）。

### 17.4 产出物与目录结构（建议统一）
```
runs/<exp_name>/
  samples/
    samples.npz
    cif/
  eval/
    tier0_metrics.json
    tier1_2d_metrics.json
    plots/
      min_dist_hist.png
      volume_hist.png
      cond_hist.png
      thickness_vacuum_scatter.png
```
- 每个指标输出 `count/median/p10/p90`，并记录 `seed/num_samples/filters` 到 metadata。

### 17.5 执行节奏（最小闭环）
- 每次采样：Tier-0/0.1 + 失败原因统计。
- 每个里程碑 checkpoint：Tier-1 + 分布对齐图。
- 对比实验/消融：固定采样规模与 seed，输出同一套评估面板。

### 17.6 验收标准（当前阶段）
- 指标输出稳定：无 NaN/Inf；同一 checkpoint 可复现。
- `valid_rate` 与 `2D_rate`（无跨真空连接）稳定提升。
- 分布统计（V/min_dist/anisotropy）不显著偏离训练集。
- 若做 recon：模型优于 `15% displacement` baseline。

注意：本节以当前 2DGEN 无条件生成任务为主，XRD 条件一致性评估保留为后续扩展项，不在当前落地范围内。
