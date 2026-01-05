# 2D-Material Canonical Representation for Transformer Diffusion Models (Scheme A+++ v3.2 Refined)

> 面向：**基于 Transformer 的扩散/Flow-Matching 晶体（2D slab）生成**  
> 目标：在 **2D-PBC（仅 XY）** 与 **z 非周期 + 真空 gauge** 的物理约束下，构建**稳定、可对齐、可逆**（round-trip）的 token 表示，并在训练/采样阶段持续强化“二维材料”的统计与几何先验。

---

## Changelog（相对 v3.1 的关键增强点）

> 注意：本节记录的是设计目标，当前代码并未全部落地；已实现能力以实际代码与 `problem.md` 为准。

1. **邻居构图策略强化 2D 先验**：引入 `kNN(d_xy)` 主图 + `kNN(d_3d)` 辅图（或补边）机制，避免 3D kNN 在多层/起伏 slab 中过度选择跨层边。  
2. **PBC wrap 信息显式编码**：将 MIC 搜索得到的整数平移 `(m,n)`（或等价的 wrap flag / shift class）做成 edge embedding，减少周期边界伪影。  
3. **排序与规约稳定性“CI 单测化”**：新增“微噪声扰动不翻转/少翻转”单测、near-degenerate cell 的 fallback/日志统计机制。  
4. **厚度/真空建模更工程化**：明确 `t` 的 4 种策略（S0–S3）的训练/采样接口与推荐默认（S1/S2）。  
5. **生成期 2D 合法域投影更严格**：明确每步投影与投影频率，避免 train/test 域不一致。

---

## 0.x 当前实现对齐状态（以仓库代码为准）

### 已对齐/已落地
- **A++ v3 预处理**：`data/preprocess.py` 实现 2D reduction、torus mean shift、z 平移与 flip、`t` 统计、`z_norm` clip、`lattice_param` (log-Cholesky)、`(Z,z_norm,u,v)` 排序。
- **缓存字段**：`prepare_c2db_tokens.py` 保存 `uvz/uv_angle/z_norm/t/lattice_param/counts_vector/order_idx` 等字段到 npz。
- **训练/采样主输入**：当前主通路使用 `Z/F/gram6`（`C2DBTokenNPZDataset` 读取 `z,f,gram6,atom_mask`）。
- **2D PBC**：默认 `pbc_mask=1,1,0`，`frac_mic_dist` + `build_knn` 计算邻居。
- **扩散目标**：连续 `F/g` v-pred + 离散 `Z` mask diffusion（`AtomVelocityLoss`）。
- **条件扩散**：`counts_vector` 为默认条件；可通过 `--cond-fields` 加入 `lattice_param/t`。
- **几何前向与损失**：`uv_angle/z_norm/lattice_param/t` 已接入模型前向与 loss。
- **双图与 wrap embedding**：支持 `--dual-graph` 与 `--wrap-embed-dim`。
- **采样期几何投影**：支持 `--project-geometry`（uv_angle/z_norm 投影）。
- **canonical 坐标缓存**：预处理可写入 `f_canon/lattice_canon/gram6_canon` 与 `order_inv`，训练可切换 `--coord-frame canon`。

### 未对齐/设计中
- **坐标系一致性与字段对齐**：`uv_angle/z_norm` 与 `z/frac` 的顺序和坐标系仍需修复（详见 `problem.md`）。

> 结论：本 guide 仍是“设计目标 + 预处理已落地 + 训练/采样部分落地”的混合文档；如需完全对齐实现，应以 `process.md` 与脚本为准。

---

## 0. 目标与基本假设

### 0.1 目标
- 输入条件：化学式/元素计数（composition），可选空间群/点群/层群等（后续扩展）。
- 输出：一个 2D slab 的 **canonical tokens**（原子序列 + 2D 晶格参数 + 厚度/真空参数）。
- 核心约束：
  - **仅 XY 做 PBC**（2D torus），**z 不做 wrap**。
  - `u,v` 必须严格定义为 canonical 的 `(â,b̂)` 基底下分数坐标。
  - 真空不学习：`c_len = t + v0`，其中 `v0` 固定或由数据统计给定。

### 0.2 基本假设（推荐先从“干净子集”开始）
- 数据是单层或少层 slab，且真空足够大，不存在 slab 与其周期像在 z 方向的相互穿越。
- CIF/结构中存在面外法向可辨识（或者 slab 明确以 z 为法向，否则需先做自动旋转对齐）。
- 若数据中存在极端 buckling/多层堆叠，请先在数据侧统计 thickness 分布并分桶训练。

---

## 1. Representation（模型接口定义）

### 1.1 Model-facing（要扩散/要输入模型的字段）
每个样本包含：

**(A) per-structure**
- `lattice_param`：2D 晶格的 SPD 参数化（推荐 log-Cholesky / 2D metric）
  - 例如：`(log_area, p1, p2)` 其中  
    `L = [[exp(p1), 0], [p2, exp(-p1)]]`，`G = exp(log_area) * L^T L`
- `t`：slab 厚度（可作为条件或预测头，默认不扩散）
- `v0`：真空厚度 gauge（常量，不扩散；导出时 `c_len = t + v0`）
- `composition`：元素计数/配比（条件输入）

**(B) per-atom（长度 N，带 mask）**
- `Z`：原子序数（离散，通常不扩散；也可用类别扩散/采样后纠错）
- `uv_angle`：`[cos(2πu), sin(2πu), cos(2πv), sin(2πv)]`
- `z_norm`：z 轴归一化坐标（连续，扩散变量之一）
- 可选：`is_surface`/`layer_id`（若你能可靠得到；否则不建议硬加）

> 当前实现：模型前向使用 `Z/F/gram6`，而不是 `uv_angle/z_norm/lattice_param`。

### 1.2 Reconstruction-only（训练可存储，但默认不扩散）
- `a_hat, b_hat`：canonical 面内基矢（由 `lattice_param` 可重建）
- `n`：slab 法向（通常固定为 z 轴；若做全姿态生成则需扩散/条件化）
- `u,v`（真实分数坐标，仅用于 debug 与 round-trip）
- `z`（Å），`z_shift`（gauge 修正量）

---

## 2. Canonicalization Pipeline（3D slab → canonical 2D tokens）

> 目标：给定任意等价的 slab 描述（平移/基底选择/轻微噪声），输出稳定一致的 tokens。

### Step 0 — dtype/单位
- Å 为长度单位；内部 float32/float64 统一。
- 所有阈值 eps 在配置里集中管理（建议默认 `eps_len=1e-6 Å`, `eps_angle=1e-6`）。

### Step 1 — slab 法向 `n`
推荐默认：
- 若数据天然沿 z 为面外方向，则 `n = (0,0,1)`。
- 否则：用 PCA / 最小方差轴 / 2D plane fit 获取 `n`，并旋转对齐到 z。
- **near-degenerate**（几乎 3D 均匀）样本直接标记/丢弃或单独训练。

### Step 2 — 投影到平面内/外
- 将 3D 晶格向量 `a,b,c` 投影到 `(I - nn^T)` 得 `a⊥, b⊥, c⊥`
- slab 的面内基矢候选一般来自 `a⊥, b⊥`（若它们线性相关则 fallback）。

### Step 3 — 投影面内基矢（推荐默认做）
- 只保留面内分量，避免原 CIF 的微倾斜导致 `u,v` 不纯。

### Step 4 — 2D reduction 去面内基矢 gauge（输出 â,b̂）
目标：在 GL(2,Z) 等价类里选唯一代表（尽量稳定）。
- V0（稳健优先）：枚举小范围 unimodular 变换 `U ∈ GL(2,Z)`（比如 |m|,|n|≤2）
- V1（快）：Minkowski reduction / “短矢优先 + 夹角范围 + 右手系”  
- tie-break（必须 deterministic）：
  1) `||â|| ≤ ||b̂||`
  2) `angle(â,b̂) ∈ [60°,120°]`（或等效约束）
  3) det>0（右手系）
  4) 若仍多解：按 `(||â||, ||b̂||, angle, a_x, a_y, b_x, b_y)` 字典序

> **工程增强（v3.2）**：  
> 对 near-degenerate cell（如近正方/近六方）添加“稳定性监控”：
> - 记录 reduction 选择的候选数量、tie-break 触发频率  
> - 若某一类结构 tie-break 频繁，优先在数据侧清洗/规约晶格输入

### Step 5 — 计算 `u,v`（严格定义）
- `u,v` 必须是 **在 (â,b̂) 基底下** 的分数坐标  
  `r_parallel = x*â + y*b̂` → `(u,v) = (x,y) mod 1`
- 绝对不要用原 CIF 的 fractional 直接当作 `u,v`（除非 CIF 已在同一 gauge 下）

### Step 6 — 去面内整体平移 gauge（torus circular mean shift）
- 在 torus 上计算 circular mean：  
  `μ_u = atan2(mean(sin 2πu), mean(cos 2πu)) / (2π)`，同理 `μ_v`
- 做平移：`u ← (u - μ_u) mod 1`，`v ← (v - μ_v) mod 1`
- 目的：使整体平移不成为学习目标，提升序列对齐稳定性

### Step 7 — 去 z 平移 gauge + slab flip gauge
- z 平移：令 `z_min = 0` 或令 `mean(z)=0`（二选一，保持一致）
- slab flip：若 `z` 的分布更靠近上表面（或按某规则），做 `z ← t - z`，保证统一朝向
- **建议**：flip 规则要只依赖 slab 内部信息，避免引入外部标注依赖

### Step 8 — 厚度 `t` 与 `z_norm`
- `t = percentile(z, 95%) - percentile(z, 5%)`（或基于原子半径修正的 robust thickness）
- `z_norm = z / t`（或 `(z - z_center)/t`，视你的 z gauge 决定）
- clip：`z_norm ∈ [0,1]` 或 `[-0.5,0.5]`（与 Step7 保持一致）
- 导出时：`c_len = t + v0`，`z = z_norm * t`（再加回 z_shift）

### Step 9 — 面内 lattice 参数化 `lattice_param`
- 由 `(â,b̂)` 得 2×2 基矩阵 `A2 = [â, b̂]`
- `G = A2^T A2`，对 `G` 做 log-Cholesky 变换得到可扩散/可回归的参数
- 采样时保证 `G` SPD（log 参数天然满足）

### Step 10 — torus angle embedding（uv_angle）
- `uv_angle = [cos(2πu), sin(2πu), cos(2πv), sin(2πv)]`
- 采样期必须投影回单位圆（见 §4.2）

### Step 11 — canonical 排序（序列对齐必需）
默认 key（建议保持与 v3.1 一致，便于兼容）：
- `(Z, z_norm, u, v)`，并对 `z_norm,u,v` 做 rounding（如 1e-5）以提高稳定性
- **v3.2 增强：排序稳定性单测（必做）**
  - 对输入原子坐标添加 1e-8~1e-7 Å 噪声，排序不应大规模翻转  
  - 若翻转集中在高对称族：考虑更强 tie-break（例如局部环境 hash）或对称等价类聚合（第二阶段）

### Step 12 — composition condition（化学式计数）
- 从 CIF/结构直接统计 `Z` 计数 → `(element_i, count_i)`
- 输入模型可用：多热向量、embedding + count、或按元素排序的稀疏字典

---

## 3. 2D PBC（XY only）与邻居几何重建

### 3.1 PBC 定义（必须写死）
- 周期性仅在 `u,v` 上：`u,v ∈ [0,1)` 的 torus
- `z` 不做 wrap（非周期）
- 任何距离、邻居、边特征都必须遵守此定义

### 3.2 MIC（默认稳健：9-candidates）
对 `Δu = u_j - u_i`、`Δv = v_j - v_i`：
- 枚举 `(m,n) ∈ {(-1,0,1)}^2`  
  `du = Δu + m`，`dv = Δv + n`  
  选择使 `||du*â + dv*b̂||` 最小的 `(m,n)`
- 返回：`du, dv, m, n`（v3.2 强烈建议保留 `m,n`）

### 3.3 3D 相对向量 `r_ij`
- `r_parallel = du*â + dv*b̂`
- `dz = (z_j - z_i)`（不 wrap）
- `r_ij = r_parallel + dz * n`（n 通常为 z 轴）
- 由此得到：
  - `d_xy = ||r_parallel||`
  - `d_3d = sqrt(d_xy^2 + dz^2)`

### 3.4 Edge features（建议）
最小可用：
- `rbf(d_3d)`（或多尺度 rbf）
- `d_xy`、`|dz|`（显式拆分，强化 2D 信息）
- `dir = r_ij / (d_3d + eps)`（可选）

### 3.5 邻居构图策略（v3.2：强化二维先验的关键）
> **核心思想**：让模型“先看到面内键合网络”，再补充必要的面外/跨层几何。

**推荐默认：双图/补边策略（二选一）**

**方案 A：双图（two-graph）**
- 主图 `E_inplane`：`k_in` 近邻基于 `d_xy`（在每个节点 i 上取最小的 k_in）
- 辅图 `E_3d`：`k_3d` 近邻基于 `d_3d`（小 k_3d，避免跨层主导）
- 模型中：
  - 要么把两类边拼接，并给一个 `edge_type` embedding（inplane / 3d）
  - 要么分两次 message passing / attention bias 叠加

**方案 B：补边（inplane kNN + 规则补充）**
- 先做 `kNN(d_xy)` 得主边集
- 再补充满足 `d_3d < r_cut_3d` 且 `|dz| < dz_cut` 的少量边（或每点补充 top-k）
- 优点：实现更简单；缺点：边类型不如双图清晰

**为什么这能强化 2D？**
- 2D 材料的主要化学键与局域环境在面内；3D kNN 在多层/强起伏 slab 中会偏向跨层近距，导致模型学到“堆叠几何”而不是“层内拓扑”。

> 当前实现：仅单图 kNN（基于 MIC 距离），未区分 d_xy/d_3d。

### 3.6 PBC wrap embedding（v3.2：减少边界伪影）
- 从 MIC 得到 `(m,n)`  
  - `(0,0)`：不跨边界  
  - 非零：跨了 u/v 方向边界
- 生成 edge feature：
  - `wrap_flag = 1[m!=0 or n!=0]`
  - `shift_id`：把 9 种 `(m,n)` 映射到 0..8 的类别 id，用 embedding
  - 或者直接把 `(m,n)` 作为小整数输入（再 MLP）

**直觉**：在 torus 上，跨边界边与非跨边界边几何等价，但在序列与有限邻域里容易产生“边界处结构断裂”。显式编码 wrap 可显著缓解。

> 当前实现：未显式编码 wrap。

---

## 4. Diffusion State & Sampling-time Manifold Projection

### 4.1 扩散变量（默认）
- `z_norm`：连续扩散变量
- `uv_angle`：连续扩散变量（在单位圆上）
- `lattice_param`：连续扩散变量（SPD via log 参数）
- `Z`：默认不扩散（条件由 composition + 采样后 assignment/纠错）；也可用离散扩散但工程复杂度更高

> 当前实现：扩散变量是 `F`（frac）与 `gram6`，`Z` 为离散 mask diffusion。

### 4.2 采样阶段必须同步的投影（否则 train/test 域不一致）
每一步（或每 1–5 步）做：

1) **uv_angle → unit circle**
- 对每个 `(cos_u, sin_u)`：  
  `r = sqrt(cos_u^2 + sin_u^2)`，归一化为 `/ max(r, eps)`
- v 同理

2) **lattice_param → SPD**
- 若用 log-Cholesky：天然 SPD；只需防止数值溢出（clip p1/p2/log_area）

3) **z_norm clip**
- `z_norm ∈ [0,1]`（或 [-0.5,0.5]），与 canonicalization 保持一致

4) **可选：最小距离软约束投影**
- 若发现采样早期出现大量近距离冲突，可在投影后做一次轻量 repulsion step（不建议过强，以免破坏扩散轨迹）

> 当前实现：仅提供 `--project-each-step`（`frac` wrap + lattice clip），不含 uv_angle/z_norm 投影。

---

## 5. 厚度 `t` 与真空 `v0` 的建模策略（工程化建议）

### S0：固定厚度（最简单）
- `t` 取数据均值/分桶均值
- 适用于：单一材料家族/厚度变化很小的数据集

### S1：`t` 作为条件输入（推荐默认）
- 模型输入：`t`（可由外部给定、或从 composition/类别先验预测）
- 扩散只生成 `z_norm`，导出时 `z = z_norm * t`
- 优点：稳定；缺点：需要 t 的来源

### S2：预测头 `t_head`（推荐）
- 网络输出一个 `t_pred`（回归），训练用 `L_t = |t_pred - t_gt|`
- 采样时用 `t_pred` 作为重建尺度
- 优点：端到端；缺点：t 学不好会放大 z 误差（需配合数据分桶/正则）

### S3：扩散 t（最难）
- 把 t 纳入扩散变量
- 仅在你有足够数据、且 thickness 分布复杂时再考虑

**真空 `v0`**
- 强烈建议常量（不扩散）
- 可取数据集统计 95% 分位的 “空隙高度” 或固定工程值（如 15–25 Å）

> 当前实现：`v0` 未进入模型或缓存；`t` 仅作为可选条件字段。

---

## 6. Training：损失与“二维先验强化”

### 6.1 基础扩散损失（示例）
- `L_uv`：对 `uv_angle` 的噪声预测损失（或 v-pred）
- `L_z`：对 `z_norm` 的噪声预测损失
- `L_lat`：对 `lattice_param` 的噪声预测损失
- `L_comp`（可选）：composition-consistency（若你同时预测 Z 分配）

### 6.2 2D 强化项（轻量但有效）
1) **最小原子间距排斥（soft repulsion）**
- 仅对 `d_3d < d_min` 的对罚项：`L_rep = Σ ReLU(d_min - d_3d)^2`
- d_min 可取元素半径启发式或统一值（如 1.0–1.2 Å）并分元素调整

2) **z 分布正则（anti-collapse）**
- 避免所有原子坍缩在同一 z：  
  例如约束 `Var(z_norm)` 在合理范围内（按族统计），或匹配训练集分布（KL/EMD）

3) **厚度一致性（若用 S2）**
- `L_t`：厚度回归损失  
- 并对 `z_norm` 的尺度不确定性做权重（例如让网络输出 `σ_t` 做不确定度加权）

---

## 7. Postprocess：tokens → CIF/结构（round-trip）

1) 由 `lattice_param` 重建 `â,b̂`
2) 由 `uv_angle` 反算 `u,v`（`atan2` / 2π，再 mod 1）
3) 由 `z_norm` 与 `t` 得 `z`
4) 构造 3D cell：
   - `a = [â,0]`，`b = [b̂,0]`
   - `c = [0,0,t+v0]`
5) 组装 fractional：
   - `frac = (u,v, z/(t+v0))`（注意 z gauge）
6) 写 CIF 前做一次合法性检查：
   - d_min 检查
   - 体积分布/面内角度检查（对 `G`）
   - thickness 合理性检查

---

## 8. 必做单测 / CI Checklist（v3.2：把稳定性钉死）

### 8.1 Canonicalization 不变性（必须）
- **平移不变**：对结构整体平移（u,v,z），输出 tokens 不变
- **面内基底等价不变**：对 `A2` 施加 `U∈GL(2,Z)`，输出 tokens 不变
- **slab flip 一致**：对 z 轴翻转，canonical 输出相同（或符号一致）

### 8.2 Round-trip（必须）
- `structure → tokens → structure'`  
  - `d_xy` 与 `d_3d` 统计一致  
  - composition 一致  
  - 若允许重排序，需满足原子集合一致（assignment 后）

### 8.3 MIC 一致性（必须）
- 9-candidates MIC 与 brute-force（更大枚举）在随机抽样上输出一致（误差 < eps）

### 8.4 排序稳定性（强烈建议）
- 添加微噪声 `1e-8~1e-7 Å`：
  - 排序翻转比例应很低（例如 < 0.5% 原子对）
  - 若高：记录族/结构类型，作为数据清洗或 tie-break 强化依据

### 8.5 near-degenerate 监控（强烈建议）
- 记录 reduction 候选数、tie-break 触发次数、PCA fallback 触发率
- 这些统计在训练前输出直方图，用于决定是否需要分桶或剔除

---

## 9. 训练与采样的推荐默认配置（可直接落地）

- Canonicalization：
  - reduction：V0（小范围 GL(2,Z) 枚举）作为默认；V1 作为加速可选
  - z gauge：`z_min=0`，`z_norm ∈ [0,1]`
  - 排序 key：`(Z, z_norm, u, v)` + rounding(1e-5)

- 图构建：
  - `k_in = 12`（按 d_xy）
  - `k_3d = 4`（按 d_3d）或补边策略
  - edge features：`rbf(d_3d) + d_xy + |dz| + wrap_shift_embed`

- 厚度策略：
  - 默认 S2（t_head）+ 失败回退到 S1（外部 t 先验/分桶均值）

- 采样投影：
  - 每步做 `uv_angle` 归一 + `z_norm` clip
  - 每 2–5 步做 lattice 参数 clip（防爆）

---

## 10. 可扩展方向（后续迭代，不影响 v3.2 落地）
- 引入 layer group / symmetry tokens（但要注意与 canonicalization 的交互）
- 多层 slab 的 layer-aware 编码（需要可靠 layer decomposition）
- 离散 Z 的生成（类别扩散/MaskGIT/自回归纠错）与 composition 对齐

---

## Appendix A：邻居构图伪代码（便于实现）

```python
# inputs: u,v,z_norm, lattice_param -> a_hat,b_hat, thickness t, normal n
# outputs: edge_index, edge_attr

# 1) MIC for all candidate pairs (i,j) in a candidate pool
#    candidate pool can be radius prefilter based on approximate metric
du, dv, m, n = mic_9_candidates(u, v, a_hat, b_hat)

r_par = du[...,None]*a_hat + dv[...,None]*b_hat
d_xy  = norm(r_par)
dz    = (z[j] - z[i])
d_3d  = sqrt(d_xy**2 + dz**2)

# 2) build edges
E_inplane = knn_by_distance(d_xy, k=k_in)
E_3d      = knn_by_distance(d_3d, k=k_3d)

edge_index = concat(E_inplane, E_3d)
edge_type  = [0]*len(E_inplane) + [1]*len(E_3d)

# 3) edge features
edge_attr = [
  rbf(d_3d),
  d_xy,
  abs(dz),
  embed_shift_id(m,n),
  embed_edge_type(edge_type),
]
```

---

## Appendix B：常见坑与规避

- **不要把原 CIF 的 fractional 当 u,v**：除非你能证明 CIF 的 cell 与你 canonical 的 (â,b̂) 完全一致。
- **不要让 z 做 PBC**：否则 slab 会自连接，训练出“假跨层键”。
- **采样不投影 = 训练白做**：uv_angle 会偏离单位圆、lattice 会离开 SPD，导致生成崩坏且难 debug。
- **排序不稳定会让 Transformer 学噪声**：先跑单测再训练，省大量时间。

---

**版本**：v3.2 (Refined)  
**生成时间**：2025-12-31
