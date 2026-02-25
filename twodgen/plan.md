# twodgen 下一步代码开发指南（面向 Phase1–Phase5 验收 + Energy-Consistent Diffusion）

> 版本：2026-02-15  
> 适用范围：当前 twodgen（2D token diffusion）端到端流水线已跑通，Phase1–Phase5 均处于 review；下一步目标是“可发表级”的**可复现基线 + MLIP/DFT 回填验证 + 能量一致性采样（Energy-guided diffusion）**。

---

## 0. 你现在“最关键的缺口”是什么？

当前项目已经具备：
- 数据清洗→token cache→split→训练→采样→评估→聚合 的完整链路
- E0–E5 的实验协议雏形与 artifacts schema
- Phase4 的 MLIP→DFT 导入导出链路（但缺 DFT 回填验证）
- Phase5 novelty/diversity/mode-collapse 工具（但缺阈值体系与 final 验收）

**下一步缺口：**
1) **冻结一个可复现、可验收的基线 checkpoint**（所有 E1/E2/E3/E4/E5 都用同一输入，减少变量）  
2) **把 Phase1/Phase2 跑到协议口径（quick/final）并通过验收门槛**  
3) **补齐“真实 DFT 回填验证”**（Phase4 从工程链路升级成论文级证据）  
4) **把 MLIP 从“后处理筛选器”升级为“采样过程的一部分”（Energy-guided diffusion）**  
5) **建立 novelty/diversity 阈值体系 + 消融矩阵 + 可复现产物导出**（支撑 npjCM 级别的论证）

---

## 1. 统一开发原则（必须遵守）

### 1.1 单一事实来源（Single Source of Truth）
- 训练与采样所有关键参数必须写入 `run_metadata.json`（或等价文件）  
- 任何 override（例如 g_scale、schedule）必须可追踪、可聚合、可复验

### 1.2 统一 run layout 与 artifacts 产物
每个 run 必须至少产出：
- `run_metadata.json`
- `projection_stats.json`
- `per_sample.jsonl`
- `metrics_summary.json`
- `failure_breakdown.json`
- `plots/`（可选但推荐）
- `samples/`（结构缓存/可选 CIF）

> 目标：最终写论文时只从这些 artifacts 生成表格与主图，不再人工拼凑。

### 1.3 可复现执行约定（uv + cache）
为避免 `/root/.cache/uv` 权限或环境不一致，统一使用：

```bash
export XDG_CACHE_HOME=/tmp/xdg-cache
export UV_CACHE_DIR=/tmp/uv-cache
2. 里程碑路线图（建议按这个顺序推进）
M0：冻结可发表基线（强制优先）
目标：得到一个“非 smoke”的 baseline checkpoint，并跑通 E1_1 quick（seeds=0/1/2）。

交付物：

baseline_ckpt_v1（带 config hash）

E1_1_quick 三个 seed 的 _aggregate/summary.json

baseline_model_card.md（训练数据、过滤条件、采样设置、失败类型占比、核心指标）

验收：

quick 口径稳定可复现（seed 间方差可接受）

artifacts schema 全部通过 validator

M1：Phase1 / Phase2 协议化验收（论文基座）
目标：把 E1/E2 在 quick/final 口径下跑完，产出稳定结论，并修复“口径无意义”的点（如 E2 schedule 轴使用同一 checkpoint）。

交付物：

E1_1_final（num_samples=20000, seeds=0/1/2）完整聚合

E1_2（组件矩阵）在中等预算下定位收益来源与副作用（特别关注 volume projection 对 diversity 的影响）

修订 E1_3 的 g_scale sweep 口径（避免用 override 强行覆盖训练时尺度）

E2_1：补齐 “不同 schedule → 不同 checkpoint” 后的回归结果

验收（建议写成 gate）：

Phase1：delta_success_geom_rate >= 0.15（final 口径）

Phase2：训练 proxy 触发率在后半程下降且几何成功率不退化；schedule 轴结果有意义

M2：Phase4 升级为论文级证据（真实 DFT 回填）
目标：把“MLIP→DFT”从“链路存在”升级为“可量化的可靠性结论”。

交付物：

dft_backfill_set_v1：从生成样本分层抽样的 DFT 回填集合（结构 + 元信息）

dft_results.jsonl：DFT relax 后的能量/结构指标

mlip_vs_dft_report.md：MLIP 误判率、排序一致性、结构偏移统计

验收：

至少完成一个可复现的 backfill protocol（例如每轮 50–200 个样本，分层抽样）

明确指出 MLIP 在 OOD 结构上的失效模式（为 M3 的 guidance/active loop 提供动机）

M3：Energy-Consistent Diffusion（核心创新点落地）
目标：实现并验证 MLIP-in-the-loop guidance：采样过程中引入能量梯度，使采样分布趋近

𝑝
𝛽
(
𝑥
∣
𝑐
)
∝
𝑝
data
(
𝑥
∣
𝑐
)
exp
⁡
(
−
𝛽
𝐸
MLIP
(
𝑥
)
)
p 
β
​
 (x∣c)∝p 
data
​
 (x∣c)exp(−βE 
MLIP
​
 (x))
交付物：

energy_guidance 模块（可插拔）

E4_energy_guidance 实验：no-guidance vs post-relax vs in-loop guidance（同预算对比）

指标：几何有效率、碰撞率、MLIP 能量分布、DFT 回填通过率提升

验收：

在同等采样预算下，in-loop guidance 对 DFT 通过率有显著提升（至少趋势明显）

不明显增加 mode collapse（与 Phase5 novelty/diversity 共同评估）

3. 代码级 ToDo（按模块拆解）
下面每一项都建议以 PR 为单位提交，并在 PR 描述里贴上 “runs/…” 的 artifacts 链接作为证据。

3.1 实验协议与 checkpoint 冻结（M0/M1）
3.1.1 增加 baseline 配置快照与 hash
改动点（建议）

在训练入口处（训练 driver / trainer）：

将完整训练配置写入 run_metadata.json

计算 config_hash（yaml/json 序列化后 hash）并写入 metadata

checkpoint 文件名或目录中包含 config_hash 的短前缀

验收

相同 config 重跑产物一致（或在可解释范围内）

3.1.2 修订 E1_3：g_scale sweep 的口径
当前问题：直接 --override-g-scale 扫 (0.5, 1.0, 1.5) 会与 checkpoint 训练时的 g_scale 数值尺度不一致。

建议改法（二选一）

A. 仅允许 relative_g_scale = {0.5,1.0,1.5}，真实 g_scale = ckpt_g_scale * relative

B. 禁止 override sweep，改为“不同 g_scale 训练得到的不同 checkpoint 对比”

代码改动（建议）

在 checkpoint metadata 中写入 train_g_scale

sampling 端读取 train_g_scale 并执行相对扫

验收

sweep 不再全失败；能解释不同 g_scale 对碰撞率/几何有效率/多样性的影响

3.1.3 Phase2：补齐“不同 schedule → 不同 checkpoint”
当前 E2_1 使用同一 checkpoint，导致 schedule 轴没有意义。

建议改动

明确训练产物命名：

CHECKPOINT_LINEAR

CHECKPOINT_SIGMOID

CHECKPOINT_COSINE

或训练脚本一次性跑三次并输出三份 checkpoint

E2_1 runner 强制检查：若 schedule 不同但 checkpoint 相同，则直接 fail-fast 并提示

验收

E2_1 的 schedule 轴出现可解释差异（至少在 proxy trigger 与几何指标上）

3.2 DFT 回填验证（M2）
3.2.1 建立 DFT backfill protocol（分层抽样）
目标：避免只验证“看起来最好”的样本。

实现建议

从 per_sample.jsonl 中读取样本指标，按 strata 分层，例如：

MLIP 能量分位数（低/中/高）

min_dist 桶

thickness/vacuum 桶

novelty 分数桶（若已有）

每个桶抽固定数量，输出 backfill_manifest.jsonl

产物

runs/<exp>/<run>/dft_backfill/backfill_manifest.jsonl

每条记录包含：sample_id、结构来源路径、关键几何指标、stratum 标签

3.2.2 DFT 作业导出与结果回收（IO 标准化）
实现建议

export_dft_jobs.py：

输入：manifest

输出：dft_jobs/<job_id>/POSCAR|.cif + INCAR/KPOINTS/POTCAR stub（视你们 DFT 工具链）

collect_dft_results.py：

解析 DFT 输出（能量、收敛、最终结构）

写 dft_results.jsonl

merge_backfill_into_artifacts.py：

将 DFT 结果按 sample_id 回填到 per_sample.jsonl 的扩展字段中（或单独生成 tier2_dft.jsonl）

验收

任意一次 backfill 都能完整复现：manifest → jobs → results → merged artifacts

3.2.3 MLIP vs DFT 分析报告（自动化）
输出至少包含：

MLIP relax 前后 vs DFT relax 后：结构偏移统计（RMSD/晶格参数偏差/键长角漂移）

能量排序一致性（Spearman/Kendall）

“MLIP 误判率”：MLIP 认为低能/稳定但 DFT 不稳定（或不收敛）的比例

失败模式归因（高 vacuum/极薄 slab/高碰撞等）

3.3 Energy-Consistent Diffusion（M3：采样期 guidance 优先落地）
先做 推理期 guidance（不改训练），最快验证增益；之后再考虑蒸馏/联合训练。

3.3.1 新增 energy guidance 配置与接口
建议新增一个配置对象（名字可自行调整）：

@dataclass
class EnergyGuidanceConfig:
    enabled: bool = False
    beta_max: float = 1.0          # 终态强度
    beta_schedule: str = "cosine"  # "linear"|"cosine"|"snr"
    start_frac: float = 0.5        # 仅在后半程启用
    every_k: int = 1               # 每 k 步施加一次
    step_size: float = 1e-3        # guidance 步长（或由 dt 缩放）
    clip_grad_norm: float = 10.0   # 防止炸
    projection_after: bool = True  # guidance 后是否投影到 2D 约束
3.3.2 新增 MLIP 统一封装层（避免采样器绑死某个模型）
建议新增：

twodgen/mlip/registry.py：统一加载 CHGNet/MACE/NequIP/…（至少先支持你们现有 CHGNet relax）

twodgen/mlip/interface.py：

energy(x) -> E

forces(x) -> -∇E

relax(x) -> x_relaxed, info

这样 sampling 里只依赖接口，不依赖具体模型。

3.3.3 在采样循环中插入 guidance（关键实现点）
你们的采样大致是：

denoiser 反向扩散更新

guardrail（vacuum/min_dist/repulsion）

projection（angle/cond/volume…）

optional relax

建议插入位置：

方案 A（推荐）：在每步 denoiser 更新后，guardrail 前插入

方案 B：在 projection 后插入（更稳定但可能抹平 projection 的意图）

guidance 更新（概念形式）：

𝑥
←
𝑥
−
𝜂
(
𝑡
)
 
𝛽
(
𝑡
)
 
∇
𝑥
𝐸
MLIP
(
𝑥
)
x←x−η(t)β(t)∇ 
x
​
 E 
MLIP
​
 (x)
实现要点：

beta(t)：建议从 start_frac 开始爬升到 beta_max

对梯度做 clip_grad_norm

guidance 后立即执行 post-projection（保持 2D slab 合法性）

3.3.4 新增实验：E4_energy_guidance（同预算公平对比）
对比三组（同 checkpoint、同 seeds、同 num_samples、同 steps）：

baseline diffusion（无 MLIP）

baseline + post-relax（你们现有 phase4 思路）

baseline + in-loop energy guidance（新方法）

必须输出：

metrics_summary.json + failure_breakdown.json

projection_stats.json

Phase5 novelty/diversity 指标

回填 DFT（至少小规模）对比通过率

验收：

组 (3) 在 DFT 通过率/几何有效率上显著优于 (1)(2)

novelty/diversity 不出现灾难性下降（mode collapse 明显上升则需要调 beta(t)/start_frac）

3.4 Phase5 阈值体系与论文级消融矩阵
3.4.1 固化 novelty/diversity/mode-collapse 的“验收阈值”
建议输出一个 thresholds.yaml：

validity 下限

novelty 下限（相对 heldout 或训练集覆盖率）

diversity 下限（例如 fingerprint 距离分布）

mode collapse 预警阈值（duplicate ratio/cluster entropy 等）

并在 validate_artifacts 中增加：

如果关键指标 < 阈值 → 标记为 FAIL 并写入 status.json

3.4.2 论文级消融矩阵（建议最少包含）
projection：volume / cond / angle / cond+angle / full

schedule：linear / sigmoid / cosine（必须对应不同 checkpoint）

repulsion：on/off

MLIP：none / post-relax / in-loop guidance

symmetry：off / soft / hard（若 phase3 要写）

每个维度至少跑 quick 口径做趋势，最终挑 2–3 个组合跑 final。

4. 推荐目录与文件组织（建议，不强制）
你们目前 twodgen/scrip/ 是兼容层，建议逐步收敛入口目录，避免后期维护成本飙升。

建议新增/调整：

twodgen/sampling/energy_guidance.py（guidance 逻辑）

twodgen/mlip/（统一接口 + registry + relax/energy/force）

twodgen/dft/（manifest → job export → result collect）

twodgen/experiments/（E1/E2/E3/E4/E5 实验矩阵的配置化描述）

configs/（baseline/train/sampling/energy_guidance/thresholds）

5. 每次提交 PR 的“必须附带证据”（强制执行）
每个 PR（除了纯重构）必须附带：

关联的 run 路径（runs/...）

_aggregate/summary.json 的关键指标对比（至少一项变好或解释清楚为何不变）

validate_artifacts 通过截图/日志片段

若改动影响协议口径：更新 process.md 中对应 Phase 状态

6. 最短可走通的执行顺序（照着做就能推进）
M0：训练 baseline_ckpt_v1 → 跑 E1_1 quick seeds=0/1/2 → 聚合通过

M1：E1_1 final → 修订 E1_3 g_scale sweep 口径 → 补齐 Phase2 三个 schedule checkpoint → 跑 E2_1

M2：建立 backfill manifest → 导出 DFT jobs → 回收结果 → 自动报告（MLIP vs DFT）

M3：实现 energy guidance → 跑 E4 对照 → 小规模 DFT 回填证明提升 → 调 beta(t) 避免 mode collapse

Phase5：固化 novelty/diversity 阈值 → validator 自动判定 PASS/FAIL → 形成论文级表格与主图

7. 你应该期待的“阶段性成果长什么样”（写论文用）
一张表：baseline vs projection vs schedule vs MLIP（validity/DFT pass/novelty/diversity）

一张图：energy-guided diffusion 的 beta(t) 设计 + 指标随 beta 的变化曲线

一张图：MLIP vs DFT 的误判率与失效模式（分层统计）

一张图：novelty/diversity 对比（证明不是“变保守”）

附录：artifact schema + 复现命令 + run layout（你们强项）

8. 附：Energy-Consistent Diffusion（实现时要记住的两条铁律）
beta(t) 不要从一开始就很大

推荐：后半程启用 + 余弦/线性爬升

guidance 之后必须回到 2D 可行域

推荐：guidance → clip → projection → guardrail → next step

到这里，这份指南已经把“下一步代码该做什么”拆成了可执行、可验收、可写论文的最小闭环。
你只需要按 M0→M3 的顺序推进，每一步都能产出实证 artifacts，最后自然拼成论文。

::contentReference[oaicite:0]{index=0}