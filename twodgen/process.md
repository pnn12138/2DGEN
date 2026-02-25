# twodgen 项目进度（当前版）

> 更新时间：2026-02-25  
> 本文件只保留“当前仍有效”的进度、架构和待办。历史复盘与已修复细节不再在此展开。

## 0. 当前目标
- 构建可复现的 2D 晶体 token diffusion 训练-采样-评估流水线。
- 用统一协议（schema + run layout）支撑 E0–E5 实验与论文产物导出。

## 1. 当前主线架构（保留）

### 1.1 数据与预处理
- 数据清洗：遍历 summary CSV，把每一行 CIF 解析成结构，基于 vacuum 轴、厚度、min_dist、cross-vacuum 和原子数等条件打标签，输出 audit/clean 表、quality bucket/hard pass 记录、quality JSONL 以及各类 failure/metrics 聚合，用于选取训练/采样行；同时保留 parse error 示例、source bucket 统计与按 fail reason 的 metric 摘要以便排查异常。
- token 化：对通过清洗的样本做 canonical normalization——用 PreprocessConfig 估计 slab 正规基、t/(uv)/z_norm、atom order、counts vector 与 lattice 参数，按 max_atoms 填充 z/f/atom_mask，附带 min_dist、vacuum、spacegroup、canonical lattice/Gram6 等 litmus，并把所有字段打包成压缩缓存，附带预处理统计与 coord frame/g_scale/schema metadata。
- 数据集加载：载入缓存后保留 z/f/atom_mask（及可选 canonical 坐标），通过 order_idx/order_inv 重新排列 atom，支持在 canonical/raw 坐标间切换，同时对 uv_angle/u/v/z_norm 做 reorder；任何 extra 张量都会复制到输出，collate 逻辑按 key 堆叠以供训练器使用。
- 划分策略：split 算法以 atom count、厚度 quantile bin 与最大元素构建 strata，在每个 stratum 内按 heldout_fraction 采样 heldout 样本，并生成 distribution checks（n atoms、元素频率、t-bin）检测 train/heldout 偏移；运行时的 subset wrapper 继承原 dataset 属性，使 train/heldout/all 视图共享 metadata。

### 1.2 训练
- 入口 driver 读取 Token dataset（含 canonical 对齐、counts vector、split 指标）并按 `quality_bucket`/`min_dist`/`collision` 条件挑选样本；支持 collision curriculum（从 non-collision 积极扩展至所有样本），可选质量标签、min_dist 过滤和 alert 阈值监控，在训练尾部产出 metrics JSONL、checkpoint 权重与 run metadata。
- 核心模型是一组 Transformer blocks 把 atom/state/lattice 表征编码为 velocity 预测：自注意力在原子与晶胞之间混合信息，MLP 输出 modulated shift/scale/gate 以控制 residual flow，`AtomDenoiser` 将这些预测送入 diffusion loss 计算中。扩散损失包括 z/f/g/uv/t/lattice/condition/vacuum/cross-vacuum/angle/composition（可选 symmetry）项，利用 uncertainty weighting、min_dist penalty、vacuum schedule 等策略保持物理一致性；tail adapter 机制可接 EGNN，cell init 动策可选 `CellNet`。
- Loss weight scheduler 维护多项物理项的权重曲线（vacuum、cross-vacuum、cond、chol、expand collision、volume、c_len、anisotropy 等），而晶格模块承担 Gram ⇄ lattice/cholesky 变换、neighbor graph 的 MIC/KNN 计算、relative bias 构造，以及 vacuum/collision 判据（如 thickness/vacuum gap/angle bounds）的封装。

#### 1.2.1 扩散生成模型：实现细节速查（当前代码即真相）

> 涉及文件：`twodgen/model/atom_denoiser.py`、`twodgen/common/atom_diffusion.py`、`twodgen/model/atom_transformer.py`、`twodgen/common/crystal.py`、`twodgen/common/projection.py`。

**(A) 统一的几何表示与张量 schema**
- 原子坐标一律使用分数坐标 `frac`（`(B, N, 3)`），并在采样/投影时 wrap 到 `[0,1)`：`frac = frac - floor(frac)`（便于 PBC + MIC 计算与稳定训练）。
- 晶格不直接扩散 3×3 矩阵，而扩散一个 6D 表示 `cell`（`(B, 6)`）：
  - `cell_rep="gram6"`：`cell` 即 Gram 6D（对称 Gram matrix 的 6 个独立分量）。
  - `cell_rep="cholesky6"`：`cell = [log(r11), log(r22), log(r33), r12, r13, r23]`，通过 `exp(log_diag)` + `G = R^T R` 保证 Gram SPD，再解码出 lattice（更“正定参数化”）。
- 物理尺度通过 `g_scale` 处理：模型内部以 `gram6_model` 工作，而物理 lattice 在需要几何计算时用 `gram6_phys = gram6_model * g_scale`（或等价的 lattice 乘 `sqrt(g_scale)`）恢复。

**(B) Backbone：`AtomTransformer`（cell token + atom tokens）**
- token 构造：
  - 原子 token：元素 `z` embedding + `frac` 的 torus/Fourier 编码（`torus_encode(frac, fourier_freqs)`）拼接后线性映射到 `embed_dim`。
  - 晶胞 token：`cell_mlp(g)` 得到 1 个 cell token，与 atom tokens 拼接成长度 `N+1` 的序列。
  - 可选几何辅助 head：`uv_angle (B,N,4)`、`z_norm (B,N)` 注入到 atom token；`lattice_param (B,3)` 与 `slab_t (B,)` 注入到 cell token（当训练/采样开启 `project_geometry` 时）。
- 注意力与邻接：
  - 使用基于 KNN 的 gather attention：对每个 atom 只 attend 到其 `k_neighbors` 个近邻 + cell token；cell token attend 到所有 atom（减少 O(N^2) 成本）。
  - 邻接与距离由 `frac` + lattice 在 MIC 下计算，距离用 RBF 展开；pair features 经过 `pair_mlp` 生成每个 head 的相对 bias。
  - 可选 `dual_graph`：同时构建 xy 图与 3D 图并合并；可选 `wrap_embed` 把 MIC shift（-1/0/1）编码成离散 wrap id；可选 `edge_type` 区分图来源并做 gating。
- 条件注入方式：
  - `t` 用 sinusoidal embedding 后经 MLP 得到 `cond_time`；
  - 可选 `cond_vec`（外部条件）经 `cond_mlp` 投影；可选 composition encoder 从 `counts_vector` 得到 `cond_comp`；
  - 三者线性相加得到 `cond`，再用于每个 block 的 AdaLN/FiLM 风格调制：生成 `shift/scale/gate` 控制 attention 与 MLP 两个 residual 分支。
- 输出 head（训练时预测的是“x0/目标变量”，loss 内部再换算为 velocity）：
  - `pred_x0_f: (B,N,3)`、`pred_x0_g: (B,6)`、`logits_z: (B,N,num_elements+1)`；
  - 若 `return_geom=True` 还输出 `uv_angle/z_norm/lattice_param/t` 的 x0 预测。

**(C) 训练目标：`AtomVelocityLoss`（diffusion/flow 两种口径）**
- 采样 `t`：`mode="diffusion"` 时用 logit-normal（`sigmoid(N(P_mean,P_std))`）偏向中后期；`mode="flow"` 则 `t~U(0,1)`。
- 构造噪声输入：
  - `frac_t`、`cell_t` 由 `(t, x0, noise)` 线性混合得到（diffusion 与 flow 的混合方向相反，代码里分别处理）。
  - 元素 `z` 走“mask language modeling”：按 `p_mask(t)` 将部分位置替换为 `mask_id`，网络只在 masked 位置做 CE；并额外做 composition consistency（expected counts vs remaining counts）。
- 速度监督：
  - 网络输出的是 `pred_x0_*`，loss 内部根据当前 `mode` 把它换算成 `pred_v_*`，与解析得到的 `v_*` 做 MSE（`f/g/uv/z_norm/lattice_param/t` 等）。
- 物理/几何正则（都在预测的 `pred_x0_f/pred_x0_g` 上计算，避免“噪声空间”歧义）：
  - vacuum / cross-vacuum：在预测结构上估计 c 轴、vacuum gap、跨 vacuum 的近邻对，并用 hinge/softplus barrier 形成惩罚；
  - angle / cond：从预测 lattice 计算角度越界率与条件数（Gram 与 lattice 两套 cond 都会记录），并对超阈值做 barrier；
  - 其他 guard：chol diag bounds、expand-collision（低 min_dist 时惩罚）、volume/c_len/anisotropy 等。
- uncertainty weighting：可选学习一组 `s_*` 自动平衡不同 loss 项的尺度（训练更稳、避免某一项 dominate）。

**(D) 采样：`AtomDenoiser.generate`（数值积分 + token reveal + guardrails）**
- 初始化：
  - `z` 从全 `mask_id` 开始；`frac`/`cell` 从高斯噪声开始（cholesky6 + `cell_init="iso"` 时用对角 log-scale 的各向同性初始化；可选 `cell_init="cellnet"` 用条件预测初值再加噪）。
  - 时间表：`diffusion` 为 `t: 0→1`，`flow` 为 `t: 1→0`；支持 `euler` 或 `heun` 两种更新。
- 主循环每步做：
  1) 用当前 `z/frac/cell/t` 预测速度并更新 `frac/cell`（可用 `neighbor_update_steps` 控制邻接缓存刷新频率）；
  2) 可选投影：`project_each_step` 做轻量投影（wrap frac +（可选）Gram cond/SPD 投影）；`post_project` 做更强的硬投影（angle/cond/inplane/area/vacuum 等，按 interval 触发并记录 stats）；
  3) 可选 schedule repair：vacuum schedule（逐步把 vacuum target 从宽松推到 `vacuum_min`）、min_dist schedule（必要时扩胞 + repulsion + 再投影）；
  4) 元素 token reveal：根据 `p_mask(t_next)` 目标保留的 mask 数，选择“置信度最高”的 mask 位置进行揭示；支持无约束采样（argmax/temperature/topk/topp）或在 `counts_vector` 约束下逐个扣减剩余配比。
- 末尾可选 `project_final` 做最终投影与统计记录；并保留 `last_project_stats` 供评估/聚合阶段写入 artifacts。

**(E) 可选模块：tail adapter / 几何 head**
- `tail_adapter="egnn"`：在 `pred_x0_f` 上再做一层轻量 EGNN 风格位移修正（以元素 embedding 与距离为输入），提升局部几何一致性；它不替代 backbone，只是一个可插拔的 tail。
- `project_geometry=True`：额外扩散并预测 `uv_angle/z_norm/lattice_param/slab_t`，用于更强的“可控几何”与投影/guardrail 协同（当前主要用于 phase3 的对称/几何控制扩展）。

### 1.3 采样
- 采样循环从定义好的 run layout（metadata、seed、checkpoint）和 heldout 条件开始，以 `AtomDenoiser` 反向扩散生成 token。每一步根据 schedule 执行 vacuum/min_dist/repulsion guardrail，必要时触发 post-projection（angle/cond/inplane/area）与 CHGNet relax，采样完成后输出 per-sample metrics、projection stats、samples 归档与 optional CIF/relax 产物。
- 产物：samples 归档、run metadata、projection stats 以及可选 CIF/relax 结果（包括 tier0/tier1/prop diagnostics）。

### 1.4 评估与实验协议
- 核心评估采用 geometry checkers：用 MIC 计算 min_dist、角度检测、Gram SPD/cond、面积/volume bounds、duplicate ratio 以及 vacuum thickness，按失败原因划分 tier0/tier1，并把 per-sample diagnostics 写入 `per_sample.jsonl`；评估结果再聚合成 metrics_summary/failure_breakdown，并与 projection stats/status 配对形成 artifacts schema。
- Tier2 及以上通过 heuristic scoring（vacuum/thickness/min_dist/cross-vacuum/valid penalty）生成性质预测分数，novelty/diversity modules 分析覆盖率与 mode collapse；validate_artifacts 复核所有 schema+status 完整性，protocol/aggregate 聚合多 run 结果以保证 E0–E5 实验协议输出一致。

## 2. 里程碑状态（E0–E5）
- `phase0`：done  
  说明：run schema、artifact validator、resume 机制已落地并实跑通过。
- `phase1`：review  
  说明：E1 链路已可端到端复现；当前需要用“同一 checkpoint + quick/final 预算”完成验收并冻结默认设置。
  - 2026-02-15：已训练基线 checkpoint（作为后续 E1/E2 的统一输入）：
    - `outputs/checkpoints/20260214_162331/atomdenoiser_best.pt`
    - `outputs/checkpoints/20260214_162331/atomdenoiser_last.pt`
  - 2026-02-14/15：用 checkpoint `outputs/checkpoints/20260214_162331/atomdenoiser_best.pt` 在 CPU 上跑了一个可承受预算的回归（N=512, steps=30, seed=0）：
    - `E1_1`：`runs/E1_1_cpu_s0_ckpt_20260214_162331/_aggregate/summary.json`，`delta_success_geom_rate_full_minus_baseline=+0.1523`（仅 1 seed + 小预算；需要按 plan 的 quick/final 预算复验）。
    - `E1_2`：`runs/E1_2_cpu_s0_ckpt_20260214_162331/_aggregate/summary.json`，`volume_only` 与 `full_projection` 表现接近；`cond_only/angle_only/cond_angle` 提升有限（提示“volume 投影”是主要增益来源，后续应重点检查其对 diversity 的副作用）。
    - `E1_3`：`runs/E1_3_cpu_s0_ckpt_20260214_162331/_aggregate/summary.json`，g_scale=0.5/1.0/1.5 全部失败（几乎全碰撞/触发率 1.0）。原因是该 sweep 目前用 `--override-g-scale` 强行覆盖训练时 g_scale=100 的 checkpoint，数值尺度不一致；需要把 sweep 范围改为围绕 checkpoint g_scale 的相对扫描，或仅用于“同 g_scale 训练的不同 checkpoint”对照。
  - 2026-02-15：尝试按 plan 的 quick 预算启动 `E1_1`（N=2000, seeds=0/1/2），但 CPU 运行耗时过长导致中断，当前进度（可用 `--resume` 继续）：
    - `runs/E1_1_ckpt_20260214_162331/baseline_seed0_n2000`：success
    - `runs/E1_1_ckpt_20260214_162331/baseline_seed1_n2000`：success
    - `runs/E1_1_ckpt_20260214_162331/baseline_seed2_n2000`：running（被中断）
- `phase2`：review  
  说明：train proxy 指标与 E2 对照脚本已就绪；需要在“不同训练 schedule 得到的不同 checkpoint”上跑回归以完成验收。
  - 2026-02-14/15：`E2_1` 在单一 checkpoint 下完成 CPU 小预算回归（N=512, steps=30, seed=0）：
    - 汇总：`runs/E2_1_cpu_s0_ckpt_20260214_162331/_aggregate/summary.json`
    - 由于 3 个 schedule 都使用了同一 checkpoint（未提供 `CHECKPOINT_LINEAR/SIGMOID/COSINE`），因此 schedule 维度结果基本一致；repulsion on/off 的差异在该预算下较小。
- `phase3`：review  
  说明：soft/hard symmetry 配置与评估字段已接通，待 final 阈值验收。
- `phase4`：review  
  说明：MLIP->DFT 筛选与导入导出链路已实现，待真实 DFT 回填验证。
- `phase5`：review  
  说明：novelty/diversity/mode-collapse 工具已实现，待覆盖率与新颖性阈值验证。

## 3. 本轮代码巡检结论（2026-02-10 / 2026-02-14）
- `uv run pytest -q`：37 passed。
- `uv run python -m compileall -q twodgen tests P_TASK/src P_TASK/scripts data/C2DB main.py`：通过。
- 模块入口体检（`python -m ... --help`，覆盖 `twodgen.scrip/*` + `twodgen.evaluate/*`）：
  - 通过：40
  - 失败：0
- 2026-02-14/15：在当前环境完成端到端复现（从数据->训练->采样/评估->聚合）：
  - 重建 token cache：`data/C2DB/cache/c2db_tokens_2d_based.npz`（saved_samples=16838，见 `data/C2DB/cache/preprocess_stats.json`）
  - 生成 split：`data/C2DB/cache/c2db_tokens_split.json`
  - 训练生成 checkpoint：`outputs/checkpoints/20260214_162331/atomdenoiser_{last,best}.pt`
  - 小预算跑通：`runs/E0_smoke`、`runs/E1_1_smoke`、`runs/E1_2_smoke`、`runs/E1_3_smoke`、`runs/E2_1_smoke`
  - 备注：当前环境直接 `uv run ...` 可能触发 `/root/.cache/uv` 权限问题；建议统一用 `XDG_CACHE_HOME=/tmp/xdg-cache UV_CACHE_DIR=/tmp/uv-cache` 执行以确保可复现。

## 4. 当前未修复问题（摘要）
- 当前未修复代码问题：无（详见 `twodgen/problem.md`）。

## 5. 非主线/待清理项（已从主流程剥离）
- `twodgen/scrip/` 视为兼容层，不作为长期唯一入口目录。
- 一次性实验脚本与临时 smoke 脚本不作为生产入口，后续统一归档。

详见：`twodgen/tm.md`。

## 6. 下一步建议（按优先级）
1. 训练一个“可用的非 smoke checkpoint”作为统一基线（用于 E1/E2/E3/E4/E5 的复跑与验收）
   - 当前 `outputs_pnn/checkpoints/*` 的 tiny+30 steps 仅用于链路自检，不具备结论意义。
   - 建议先跑一个可控预算的基线（例如：固定 `max_steps`、固定 seed、固定 npz/split/过滤条件），并将该 checkpoint 作为后续所有实验的唯一输入以减少变量。
2. phase1 验收顺序（先验收门槛，再定位原因）
   - 优先把 `E1_1` 跑到 plan 的 quick/final 预算并完成验收（建议在 GPU 上跑；CPU 上可先降预算做 sanity）。
   - 先跑 `E1_1` 的 final 口径（`protocol=final`、`num_samples=20000`、seeds=0/1/2）并检查是否满足 `delta_success_geom_rate>=0.15`；
   - 若仍未达标，再用 `E1_2`（组件矩阵）+ `E1_3`（g_scale sweep）在中等预算（>=2048）下定位主要收益来源与副作用，再回到 final 复跑最优组合。
   - `E1_3` 的 g_scale sweep 需先修订口径：避免在不同数值尺度的 checkpoint 上直接 `--override-g-scale` 扫 (0.5,1.0,1.5)。
3. phase2 验收（训练-采样协同）
   - 先补齐“不同 schedule 对应不同 checkpoint”的训练产物（或设置 `CHECKPOINT_LINEAR/SIGMOID/COSINE`），否则 `E2_1` 的 schedule 轴没有意义。
   - 跑 `E2_1`（schedule × repulsion）中等预算回归，并配合 `check_trigger_trend.py` 检查训练 proxy 触发率后半程下降且几何成功率不退化。
4. 工程维护（不影响实验但降低后续成本）
   - 统一 `_parse_pbc_mask` 到 `twodgen/common`，消除多处重复实现；
   - 收敛入口目录（`scripts`/`scrip` 二选一），设置 deprecation 窗口并逐步移除冗余 shim。

## 7. 维护约定
- 发现新代码问题：追加到 `twodgen/problem.md`（只保留未修复项）。
- 发现冗余代码/文件：追加到 `twodgen/tm.md`。
- 每次阶段推进后，仅更新本文件的“当前状态”，不回填大段历史流水。
