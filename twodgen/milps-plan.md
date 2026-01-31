# 基于 Transformer/EGNN 的二维晶体扩散生成：改进规划与实施方案

**版本**: 1.0
**日期**: 2026-01-29
**当前状态**: 模型训练流程跑通，但存在严重的晶格塌缩、原子碰撞（碰撞率 >90%）和二维结构失效（跨真空率 >80%）问题。条件控制（成分/对称性）尚未精准生效。
**目标**: 通过引入 MLIP 辅助、架构升级和损失函数优化，构建一个能生成高稳定性、无碰撞且满足特定条件（成分/对称性）的二维材料生成模型。

---

## 1. 核心瓶颈与对策概览

| 瓶颈现象 | 根本原因 | 改进对策 (Short-term -> Long-term) |
| :--- | :--- | :--- |
| **晶格塌缩** <br> (Volume ~71 Å³) | 缺乏显式体积约束；参数化边界截断导致梯度失效；模型走“捷径”降低重建损失。 | 1. **后处理弛豫** (Phase 1): 使用 MLIP 修复。 <br> 2. **采样约束** (Phase 2): 强制重置真空层。 <br> 3. **架构升级** (Phase 3): 独立晶胞生成网络 (CellNet) 或 EGNN。 |
| **原子碰撞** <br> (Collision Rate >90%) | 缺乏硬核排斥势；训练阶段 `min_dist` 权重过低且仅针对单对原子。 | 1. **MLIP 弛豫** (Phase 1): 直接物理优化消除重叠。 <br> 2. **多对碰撞惩罚** (Phase 2): 损失函数升级为累积惩罚。 <br> 3. **软硬结合排斥** (Phase 3): 动态排斥势。 |
| **真空层失效** <br> (Cross Vacuum >80%) | 真空权重 (`lambda_vacuum`) 为 0；缺乏对 Z 轴分布的几何引导。 | 1. **几何约束采样** (Phase 1): 强制限制原子 Z 坐标范围。 <br> 2. **分段真空损失** (Phase 2): 引入 hinge loss 惩罚小真空。 |
| **条件失效** <br> (成分/对称性) | 条件注入方式单一（仅 Input Embedding）；缺乏对齐机制；训练未应用 Dropout/CFG。 | 1. **CFG 采样** (Phase 2): 训练 Dropout + 推理 Guidance。 <br> 2. **Adapter 微调** (Phase 3): 针对特定空间群/成分微调 Adapter。 |

---

## 2. 阶段性实施计划

- ### Phase 1: 物理有效性修复 (The "Fixer" Phase)
- **目标**: 不重新训练模型，通过引入外部物理知识（MLIPs）和几何规则，快速将生成有效率 (Valid Rate) 提升至 90% 以上。

- [x] **集成 MLIP 后处理 (Post-Processing Relaxation)**  
    - **工具**: `chgnet` + `ase`，已在 `twodgen/scrip/sample_tokens.py --relax` 中集成 CHGNet-BFGS 套件，并将 `energy_mlip/relaxed_flag/min_dist_relax` 回写到 `samples.npz`。  
    - **验证任务**: 使用 checkpoint `outputs/checkpoints/20260122_134725/atomdenoiser_last.pt` 运行 `sample_tokens`（`--npz data/C2DB/cache/c2db_tokens_2d_based.npz` + `--unsafe-load`）生成 `outputs/samples_tokens/20260122_134725_smoke`，并确认 `eval/` 文件夹生成。  

- [x] **几何引导采样 (Geometric Guidance)**  
    - **逻辑**: 每步 Euler/Heun 采样通过 `AtomDenoiser._z_clamp_step` 限制 `z` 坐标，配合 `--z-clamp` CLI；`sample_tokens` 也会在 `expand_vacuum`/`min_dist_project` 中利用相同机制。  
    - **验证任务**: 对照 `min_dist_pre/post` stats（命令输出中为 0.236/0.933，0.704/1.199），确保 `z_clamp`/`repulsion` 触发并在 `samples.npz` 的 `min_dist_pre/post` 中体现。  

- [x] **建立 Tier-0/1 评估基准**  
    - **指标**: collision rate、valid_2d、min_dist、vacuum/cross_vacuum 通过 `eval_samples.py` 评估。  
    - **工具**: `chgnet` 或 `mattersim`。
    - **逻辑**: `Raw Generation` -> `Force Vacuum Expansion` (Z > 20Å) -> `MLIP Relaxation` (BFGS)。
    - **预期**: 彻底消除原子碰撞和晶格塌缩带来的非物理结构。
- [ ] **几何引导采样 (Geometric Guidance)**
    - **逻辑**: 在扩散采样循环中（Euler/Heun），每一步强制将原子 Z 坐标限制在 `[0.5-t/2, 0.5+t/2]` 区间。
    - **预期**: 解决跨真空成键 (Cross Vacuum) 问题。
- [ ] **建立 Tier-0/1 评估基准**
    - **指标**: 实现 Collision Rate, Valid 2D Rate, Formation Energy (需预算元素参考能) 的自动化计算。

### Phase 2: 模型内建约束优化 (The "Constraint" Phase)
**目标**: 修改损失函数和训练策略，让模型“学会”物理规则，减少对后处理的依赖。

- [ ] **真空与体积损失唤醒**
    - **操作**: 将 `lambda_vacuum` 从 0 调大。
    - **设计**: 引入 `Hinge Loss`: `Loss = max(0, 15.0 - c_axis) * weight`，迫使模型拉大 c 轴。
- [ ] **多对碰撞惩罚 (Multi-pair Repulsion)**
    - **操作**: 修改 `min_dist_loss`，不再只惩罚最近的一对原子，而是惩罚所有距离小于阈值 (如 2.0Å) 的原子对。
    - **调度**: 采用余弦退火策略，训练后期逐步增大此权重。
- [ ] **条件控制增强 (CFG)**
    - **操作**: 训练时以 15% 概率将 `counts_vector` 置零 (Dropout)。
    - **采样**: 实现 Classifier-Free Guidance，通过 `scale > 1.0` 强化成分控制。

### Phase 3: 架构升级与深度控制 (The "Architecture" Phase)
**目标**: 引入等变性与模块化设计，实现对复杂条件（如空间群）的精准控制。

- [ ] **引入 EGNN (Equivariant GNN)**
    - **方案**: 在 Transformer 层后接入 EGNN 模块，利用相对坐标消息传递来修正局部几何，天然保证旋转平移不变性，缓解晶格尺度敏感问题。
- [ ] **晶胞生成解耦 (CellNet)**
    - **方案**: 训练一个独立的小型网络（VAE/Diffusion）专门生成晶格参数 ($a, b, c, \alpha, \beta, \gamma$)。
    - **流程**: `Condition` -> `CellNet` -> `Lattice` -> `Atom Diffusion` -> `Structure`。这能从源头保证晶格不塌缩。
- [ ] **Adapter 微调机制**
    - **方案**: 借鉴 MatterGen，冻结主模型，针对特定空间群或特殊性质（如磁性）训练轻量级 Adapter。
- [ ] **能量引导扩散 (Energy Guidance)**
    - **方案**: 在采样过程中，利用 MatterSim/CHGNet 计算能量梯度 $\nabla E$，作为额外的 Score Term 引导生成方向。

---

## 3. 评估体系与指标对齐 (Metrics Alignment)

为了与业界先进方法 (MatterGen, PXRDGen) 对齐，我们将采用以下分层评估指标：

| 维度 | 指标名称 | 定义/计算方法 | 目标值 (Phase 1/2/3) |
| :--- | :--- | :--- | :--- |
| **几何** | **Collision Rate** | 任意原子间距 < 0.7Å 的样本比例 | < 5% / < 1% / < 0.1% |
| **几何** | **Valid 2D Rate** | 无碰撞且无跨真空成键的样本比例 | > 80% / > 90% / > 95% |
| **稳定** | **Stable Rate** | $E_{hull} \le 50$ meV/atom (需 MLIP 计算) | - / > 30% / > 50% |
| **条件** | **Comp. Match** | 生成成分与输入条件的精确匹配率 | > 80% (with CFG) |
| **多样** | **Uniqueness** | 生成结构指纹去重后的比例 | > 90% |

---

## 4. 关键技术选型

* **MLIP 模型**: **CHGNet** (首选，速度精度平衡，支持磁性) / **MatterSim** (备选，精度更高)。
* **对称性工具**: `spglib` (用于评估和硬约束)。
* **开发框架**: `PyTorch` + `ASE` + `Pymatgen`。

---
