# 2D Crystal Transformer Diffusion — Preprocessing Spec（A++ v3）

> 面向二维材料（slab）的“可复现、可单测、可缓存”预处理细化文档（基于你当前 A++ 规划，重点细化预处理部分）。

---

## 优化规划（先预处理）
### 阶段目标
1) **预处理模块更迭（当前优先级）**  
   - 把预处理流程固化为“输入/输出”的可单测函数链  
   - 统一 gauge 处理与 canonical 排序，确保缓存可复用  
   - 完成缓存格式与版本号（`A++_v3`）的定义  
2) **模型输入一致性**  
   - 训练与采样阶段的流形投影一致化（uv_angle、SPD、z_norm）  
   - 与缓存字段对齐，避免在线重复计算  
3) **评估与可视化**  
   - 增加几何一致性与 gauge 不变性回归测试  
   - 形成固定的对比图与统计输出（保存在 `test_fig/`）

### 预处理模块更迭交付物
- `prepare_*` 流程拆解为可复用函数（含单测入口）  
- 缓存字段与哈希规则落地  
- 关键超参集中管理（eps、round_prec、z_norm_clip）  

## 0. 预处理要达成什么
### 0.1 四个硬目标
1) **几何一致性**：编码得到的 `(u,v,z_norm,lattice_param)` 必须能无歧义重建 `r_ij`。  
2) **去 gauge**：vacuum / slab flip / (a,b)基矢 / 面内平移 / z 平移。  
3) **确定性**：同一输入结构 → 同一 tokens、同一排序（允许极小数值容差）。  
4) **可缓存**：预处理一次，训练反复读取，不再做 reduction / 投影 / 求逆。

### 0.2 每个样本最终输出（建议最小集合）
- `Z: int[N]`  
- `u,v: float[N]`（**在 (â,b̂) 基底下的分数坐标**，已 wrap 到 `[0,1)`）  
- `z_norm: float[N]`  
- `t: float`（厚度，Å）  
- `(a_hat,b_hat,n): float[3]`（规约后的面内基矢 + 单位法向）  
- `lattice_param: float[3]`（3 DoF；建议 log-area + shape 两参）  
- `counts_vector`（单位晶胞元素计数向量；`N = Σ counts`）  
- `order_idx`（canonical 排序索引映射）  
- （可选）`neighbor_graph`（edge_index + edge_attr）

---

## 1. 从 CIF 取“化学式/计数”与原子列表：最稳做法
### 1.1 “化学式是否从 CIF 取更好？”
更稳的不是 CIF 里的字符串字段，而是：  
**直接从 CIF 的原子位点（最终展开后的原子列表）统计 counts**。

原因：
- `_chemical_formula_sum` 可能是 reduced formula（只给比例）或忽略 occupancy  
- `atom_site`（或结构对象中的 sites）才与你真正生成的晶胞一致

### 1.2 推荐输入优先级
1) **直接已有原子列表**：`cell + pos_cart + atomic_numbers`（最佳，最少坑）  
2) **CIF**：用 pymatgen/ase 解析成结构对象后再导出原子列表（注意 occupancy 与对称展开）

### 1.3 occupancy / disorder 的策略（强烈建议先过滤）
如果你现在目标是“结构几何生成”，建议：
- **过滤掉 occupancy≠1 或无序混占**的结构  
否则任务会升级成“位点占位概率生成 + 几何生成”，难度陡增。

---

## 2. 预处理主流水线（逐步、可单测）
下面每一步尽量写成“输入/输出”，便于你拆函数与写单测。

---

### Step 0：统一 dtype / 单位
**输入**：`cell(3×3, Å), pos_cart(N×3, Å), Z(N)`  
**输出**：float64 的 `cell/pos` + int64 的 `Z`  
> 预处理建议 float64，训练再转 float32。

---

### Step 1：确定 slab 法向 `n`
#### 1.1 默认法向
- `n_raw = a × b`
- `n = n_raw / ||n_raw||`

#### 1.2 PCA fallback（只在必要时用）
触发条件（建议满足任一就 fallback）：
- `||a×b|| < eps_area`（如 `1e-6`）→ 面内退化  
- 或者你检测到 plane/lattice 明显不一致（少见，但可加）

PCA 做法：
- `X = pos - mean(pos)`
- SVD/PCA 得到最小方差方向 `n_pca`
- 方向统一：让 `sign(n_pca·(a×b))>0`，否则 `n_pca ← -n_pca`

**关键约束**：一旦用了 PCA 的 `n`，后面 **必须把 a,b 投影到该平面** 再做 reduction（否则 plane / lattice / uv 不共面）。

---

### Step 2：投影到平面内/外
对每个原子 `r_i`：
- `z_i = r_i · n`
- `r_parallel_i = r_i − z_i * n`

输出：`z(N)`, `r_parallel(N×3)`

---

### Step 3：把 a,b 投影到 slab 平面（建议默认做）
- `a_in = a − (a·n)n`
- `b_in = b − (b·n)n`

> 这一步能显著降低数值不一致风险，建议无论是否 PCA 都做。

---

### Step 4：2D reduction 得到 `(â,b̂)`（去面内基矢 gauge）
工程上建议：**先 V0（枚举）跑通；再 V1（Minkowski）提质量/提速**。

#### 4.1 V0：有限 unimodular 枚举（简单可靠）
枚举 2×2 整数矩阵 `U=[[p,q],[r,s]]`，满足 `det(U)=±1`（范围可取 `{-1,0,1}` 或更小集合），对每个候选：
- `[a', b'] = [a_in, b_in] @ U`
- 评分函数（越小越好）：
  - 惩罚长基矢：`|a'| + |b'|`
  - 惩罚接近共线：`|cos γ'|`（γ'=夹角）
  - 约束：`|a'| ≤ |b'|` 且 `γ' ∈ [60°,120°]`
取最优 `(â,b̂)=(a',b')`

#### 4.2 V1：2D Minkowski reduction（更“唯一”）
如果你希望 MIC 用 `round` 简化更安全，建议上 Minkowski。  
（但即使上了，也建议保留 9-candidates MIC 做兜底单测。）

#### 4.3 orientation 固定（避免额外等价）
- `vol = det([â, b̂, n])`
- 若 `vol < 0`：令 `b̂ ← -b̂`（或交换+修正），统一右手系

---

### Step 5：计算 `u,v`（**必须是 (â,b̂) 基下分数坐标**）
构造 `A = [â  b̂]` 为 3×2（列向量），对每个原子解最小二乘：
- `x = argmin ||A x − r_parallel_i||²`
- 闭式：`x = (AᵀA)^{-1} Aᵀ r_parallel_i`
- `u_i=x0, v_i=x1`

wrap：
- `u_i ← u_i − floor(u_i)`
- `v_i ← v_i − floor(v_i)`

数值建议：
- 对 `AᵀA` 加 `eps*I`（如 `1e-12`）防奇异  
- condition number 太大时用 `lstsq/SVD` 回退

> 这一步是 A++ 的“核心修复点”：确保你后续 `Δr_parallel = Δu*â + Δv*b̂` 一定成立。

---

### Step 6：去面内整体平移 gauge（torus circular mean shift）
计算 circular mean：
- `ū = atan2(mean(sin 2πu), mean(cos 2πu)) / (2π)`
- `v̄ = atan2(mean(sin 2πv), mean(cos 2πv)) / (2π)`

shift：
- `u ← (u − ū) mod 1`
- `v ← (v − v̄) mod 1`

> 对 Transformer（序列）非常关键：减少同一结构多表示、减少排序抖动。

---

### Step 7：去 z 平移 + slab 翻转 gauge
z 平移：
- `z ← z − mean(z)`

slab 翻转（canonical sign）：
- `m1 = Σ Z_i z_i`
- 若 `m1 < 0`：`z ← −z` 且 `n ← −n`

tie-break（近对称时）：
- 用 `m3 = Σ Z_i z_i^3` 决定符号

---

### Step 8：厚度 `t` 与 `z_norm`
厚度（鲁棒）：
- `t = q99(z) − q01(z)`（Å）

归一化：
- `z_norm = z / (t + eps)`

clip（建议）：
- `z_norm ∈ [-1.5, 1.5]`（或按你数据统计调整）

---

### Step 9：lattice_param（3 DoF，扩散友好）
构造 2D metric：
- `G2D = [[â·â, â·b̂],[â·b̂, b̂·b̂]]`

面积：
- `area = sqrt(det(G2D))`
- `log_area = log(area)`

shape：
- `G_shape = G2D / area`

推荐 **log-Cholesky**（最稳）：
- `L = chol(G_shape)`（下三角）
- `p1 = log(L11)`
- `p2 = L21`（可选 tanh 压缩）

输出：
- `lattice_param = (log_area, p1, p2)`

---

### Step 10：uv_angle（torus embedding）
- `uv_angle = (cos 2πu, sin 2πu, cos 2πv, sin 2πv)` ∈ ℝ⁴

---

### Step 11：canonical 排序（保证序列对齐）
排序前必须完成 Step 6（面内平移 gauge），否则排序对平移高度敏感。

推荐排序键（带容差离散化降低浮点抖动）：
- `z_key = round(z_norm, 1e-6)`
- `u_key = round(u, 1e-6)`
- `v_key = round(v, 1e-6)`
- key = `(Z, z_key, u_key, v_key, original_index)`

输出 `order_idx`，并重排所有 per-atom 张量。

---

### Step 12：condition（单位晶胞元素计数向量）
从 `Z` 统计 counts：
- `counts[element] = count(Z==element)`
- `N = sum(counts)`（可显式存）

> 这就是“从 CIF 得到的化学式（单位晶胞计数）”：最稳、无歧义。

---

## 3. 邻居图（可缓存/可在线）
### 3.1 MIC（默认稳健：9-candidates）
对 `(i,j)`：
- `du0 = u_j − u_i`, `dv0 = v_j − v_i`
- 枚举 `(m,n)∈{-1,0,1}²`：
  - `du = du0 − m`, `dv = dv0 − n`
  - `Δr_parallel = du*â + dv*b̂`
取 `||Δr_parallel||` 最小者作为 MIC。

> 若你未来严格保证 Minkowski reduction 并单测证明 `round` 永不失败，再把 MIC 简化为 `du0-round(du0)` 才安全。

### 3.2 3D 相对向量
- `Δz = (z_norm_j − z_norm_i) * t`
- `r_ij = Δr_parallel + Δz * n`
- `d_ij = ||r_ij||`

---

## 4. 采样阶段必须同步的“流形投影”（非常重要）
不做这一步，训练/采样域会不一致，采样很容易崩。

每一步（或每几步）强制：
1) **uv_angle 单位圆投影**：对每对 `(cos,sin)` 做归一化  
2) **SPD 保证**：如果用 log-Cholesky 参数化，天然 SPD；否则需投影到 SPD  
3) **z_norm clip**：保持在训练分布范围

---

## 5. 缓存与哈希（确保可复现）
### 5.1 建议缓存内容
- `Z, uv_angle, z_norm, lattice_param, t`
- `(a_hat,b_hat,n)`
- `order_idx`
- `counts_vector`
- （可选）`neighbor_graph`

### 5.2 样本哈希建议包含
- 原始 CIF 内容（或 cell+pos+Z bytes）
- 预处理版本号（如 `A++_v3`）
- 关键超参（eps、round 精度、reduction 版本）

---

## 6. 单测清单（预处理必须先过）
### 6.1 几何一致性（最关键）
- 六方/斜晶（γ=60°）下：`||Δr_parallel||` 与 brute-force 镜像搜索一致  
- `9-candidates MIC` 与 `round MIC` 对比：一旦存在反例，默认用 9-candidates

### 6.2 gauge 不变性
- 面内平移：整体平移 `αâ+βb̂` 后 tokens 不变  
- slab 翻转：z→−z，canonical 后 tokens 不变  
- vacuum：改变 c（加真空）tokens 不变（t 不变，c 仅用于重建）

### 6.3 排序稳定性
- pos 加 1e-8~1e-7 Å 噪声：排序变化应极少（同元素内允许小范围交换）

---

## 7. 推荐的默认超参（先跑通用）
- `eps_area = 1e-6`
- `eps_inv = 1e-12`
- `round_prec = 1e-6`
- `z_norm_clip = 1.5`
- `vacuum_v0 = 15 Å`
- `neighbor_k = 16`
- `MIC = 9-candidates`（默认）
