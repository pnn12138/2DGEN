# twodgen 项目进度（当前版）

> 更新时间：2026-02-14  
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
  说明：E1_1 quick 已完成，`delta_success_geom_rate=+0.1133`，尚未达到 `+0.15` 目标。
- `phase2`：review  
  说明：train proxy 指标与 E2 对照脚本已就绪，等待真实长程 run 验证趋势。
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
- 2026-02-14：在当前环境完成端到端 smoke 复现（从数据->训练->采样/评估->聚合）：
  - 重建 token cache：`data/C2DB/cache/c2db_tokens_2d_based.npz`（limit=5000）
  - 小跑训练生成 checkpoint：`outputs_pnn/checkpoints/<STAMP>/atomdenoiser_{last,best}.pt`
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
   - 先跑 `E1_1` 的 final 口径（`protocol=final`、`num_samples=20000`、seeds=0/1/2）并检查是否满足 `delta_success_geom_rate>=0.15`；
   - 若仍未达标，再用 `E1_2`（组件矩阵）+ `E1_3`（g_scale sweep）在中等预算（>=2048）下定位主要收益来源与副作用，再回到 final 复跑最优组合。
3. phase2 验收（训练-采样协同）
   - 跑 `E2_1`（schedule × repulsion）中等预算回归，并配合 `check_trigger_trend.py` 检查训练 proxy 触发率后半程下降且几何成功率不退化。
4. 工程维护（不影响实验但降低后续成本）
   - 统一 `_parse_pbc_mask` 到 `twodgen/common`，消除多处重复实现；
   - 收敛入口目录（`scripts`/`scrip` 二选一），设置 deprecation 窗口并逐步移除冗余 shim。

## 7. 维护约定
- 发现新代码问题：追加到 `twodgen/problem.md`（只保留未修复项）。
- 发现冗余代码/文件：追加到 `twodgen/tm.md`。
- 每次阶段推进后，仅更新本文件的“当前状态”，不回填大段历史流水。
