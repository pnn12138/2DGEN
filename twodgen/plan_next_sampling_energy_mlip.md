# twodgen 下一步规划（规划层）：采样端硬约束投影 + 能量评估链路补齐 + 可选 MLIP 微调  
**版本**：v1.0  
**日期**：2026-02-06  
**最后更新**：2026-02-06  
**范围**：本计划聚焦三条主线：  
1) 采样端加硬约束/投影兜底（angle / cond / in-plane 退化）；  
2) 补齐能量评估链路（run_metadata + success_geom/success_energy）；  
3) 可选加速项：层状材料小规模 MLIP（如 CHGNet）微调，用收益判定是否替换线上权重。  

> 说明：本计划不做训练侧“volume/scale”优化；但采样端允许加入 **volume clamp 作为硬护栏**（目标是压制 `bad_volume` 统治失败原因，优先保证闭环可验证）。

## 0.3 执行记录（2026-02-06）
- Workstream A/B/D/E 已落地（投影兜底 + 评估 taxonomy + 一键脚本 + 最小测试）。
- A/B（projection off vs on+volume clamp）已跑通：见 `outputs/ab_proj_phase2_v8`
  - A：`success_geom_rate=0.133`，`bad_volume=98/128`
  - B：`success_geom_rate=0.484`，`bad_volume=47/128`，`post_project_trigger_any_rate=0.742`，collision 19→20（对 volume 缩放样本追加更强 min_dist repulsion）
- 能量链路（CHGNet relax + taxonomy）已跑通：见 `outputs/eval_energy_phase2/eval`
  - `success_geom_rate=0.5625`，`energy_available_rate=0.125`（relax 成功样本），`fail_reason_energy_counts` 可解释（missing）。

## 0.4 完成状态（截至 2026-02-06）
- [x] Workstream A（采样端投影兜底）：`twodgen/common/projection.py` + `twodgen/model/atom_denoiser.py` + `twodgen/scrip/sample_tokens.py`
- [x] Workstream B（能量链路 + taxonomy）：`twodgen/evaluate/eval_samples.py` + `run_metadata.json/projection_stats.json`
- [x] Workstream D（A/B 回归脚本）：`twodgen/scrip/sampling_projection_ab.sh`
- [x] Workstream E（最小测试）：`tests/test_sampling_projection.py`、`tests/test_energy_chain.py`（`uv run pytest` 通过）
- [~] Workstream C（可选：MLIP 微调）：已补齐“产物与 go/no-go 模板”，但未做 finetune 训练（按计划允许后置）
  - template: `twodgen/mlip_finetune_report.md`
  - template: `twodgen/mlip_finetune_config.yaml`
  - registry stub: `twodgen/model_registry.json`

---

## 0. 目标与验收总览
### 0.1 总目标
- **几何层**：显著减少 eval 因 `angle_out_of_range / cond_violation / in_plane_degenerate` 直接淘汰，使 `success_geom` 抬升，且失败原因发生“可解释迁移”。  
- **能量层**：能量不可用/relax 失败不再是黑箱；每个样本明确输出 `energy_available`、`success_energy` 及失败原因。  
- **加速项**：若 MLIP 微调能在 *relax 成功率/力误差/能量稳定性* 上带来确定收益，再考虑替换线上权重（否则仅作为可选组件或停止）。

### 0.2 核心指标（建议统一输出到 tier0/tier1）
**采样/评估（每次采样 run）**
- `success_geom_rate`、`valid_rate`
- `angle_out_of_range_rate`
- `cond_lattice_violation_rate`、`cond_lattice_p95/max`
- `inplane_degen_rate`
- `collision_rate`
- `project_trigger_any_rate`、`project_delta_norm_p95`（仅 projection on 时）

**能量评估**
- `energy_available_rate`
- `success_energy_rate`
- `fail_reason_energy` 计数分布（missing / load_fail / runtime_error / non_converge / nan）

**MLIP 微调（如启用）**
- `force_MAE`（val/test）、`energy_MAE`（可选）、`relax_success_rate`（val/test）
- 推理/relax 时间统计（粗粒度即可）

---

## 1. Workstream A：采样端硬约束/投影兜底（P2-MVP）
### A1. 设计原则
- **投影是兜底护栏**：先“能跑通/能抬升 success_geom”，再逐步降低投影触发率（让模型学会而不是全靠护栏）。  
- **与 eval 口径一致**：投影触发条件尽量复用 eval 的判定函数/阈值，避免“采样修了但 eval 仍判失败”。  
- **最小改动**：投影尽量做“局部/连续”修正，减少对分布的破坏，并记录修正幅度。

### A2. 实施位置与开关
- 在 sampler 主循环（更新 lattice/gram6 后）加入 `post_step_project()`：  
  - CLI：`--post-project`、`--post-project-interval {0,1,5,10}`、`--post-project-keys angle,cond,inplane,volume`、`--post-project-v-min/--post-project-v-max`  
  - 默认建议：先用 **final-only**（`--post-project-interval 0`）跑通闭环，再视需要改为 interval=1。

### A3. 四类硬约束（建议顺序：angle → in-plane → cond → volume）
#### A3.1 Angle clamp（角度约束）
- **触发**：任一角 `alpha/beta/gamma` 超出合法区间（建议沿用 eval 的 angle bounds）  
- **动作**：直接 clamp 到边界或使用有界映射（采样端允许硬 clamp）  
- **统计**：  
  - `angle_oob_rate_before/after`  
  - `angle_project_trigger_rate`  
  - `angle_project_delta_mean/p95`

#### A3.2 Cond clamp（条件数约束）
- **触发**：`cond > cond_max_eval`（与 eval 同口径；2D slab 默认用 **in-plane Gram cond**，避免真空轴主导）  
- **动作（推荐方案）**：对 lattice 做数值稳定修正（目标：降低 cond，同时尽量保持方向/体积）  
  - 选项 1（推荐）：SVD/特征值裁剪：强制 `sigma_min >= sigma_max/cond_max` 再重构  
  - 选项 2：对 Gram/Cholesky 做投影到可行域（实现更复杂，但更贴近训练表征）  
- **统计**：  
  - `cond_lattice_p95/max_before/after`  
  - `cond_project_trigger_rate`  
  - `cond_project_delta_mean/p95`

#### A3.3 In-plane 退化 clamp（2D 关键护栏）
- **触发（建议至少包含）**：  
  - `a < a_min` 或 `b < b_min`  
  - `sin(gamma)` 太小（gamma → 0/180）  
  - `area_inplane = a*b*sin(gamma) < area_min`  
- **动作**：优先仅修正 in-plane：  
  - 放大 a/b 到最小值；将 gamma 拉回安全区间；必要时微调 a/b 比例但避免大幅扭曲  
- **统计**：  
  - `inplane_degen_rate_before/after`  
  - `inplane_project_trigger_rate`  
  - `inplane_area_before/after_p50/p95`

#### A3.4 Volume clamp（闭环兜底：抑制 bad_volume）
- **触发**：volume 超出训练数据 `p1/p99` 的 `[v_min, v_max]`（由 `--npz` 自动读取，或手动指定 `--post-project-v-min/--post-project-v-max`）  
- **动作**：只缩放 in-plane 两个 lattice 向量（volume 变化为 `s^2`），不动真空轴；缩放下界需保证不破坏 `inplane_a/b/area_min`。  
- **统计**：  
  - `post_project_vol_scaled_rate`  
  - `post_project_vol_before/after` 与 `post_project_vol_scale_inplane` 分布

### A4. 投影总统计（必须）
- `project_trigger_any_rate`（任意投影触发率）  
- `project_delta_norm_mean/p95`（投影改动幅度）  
- `project_steps_per_sample_mean/p95`（可选：每个样本触发次数）

### A5. 最小对照实验（AB）
- A/B：相同采样配置下  
  - A：projection off  
  - B：projection on（建议先 interval=0 只做 final；必要时再切 interval=1）  
- 只看 6 项：`success_geom_rate / angle_oob / cond_violation / inplane_degen / collision / project_trigger_any`  
- **验收口径**：B 必须显著降低 `angle_oob/cond_violation/inplane_degen`，并使 `success_geom_rate` 上升；若 collision 显著上升，需调整投影顺序/幅度。

### A6. 交付物
- `twodgen/scrip/sampling_projection_ab.sh`：一键跑 A/B  
- `twodgen/common/projection.py`（或等价模块）：`project_lattice()`、统计函数  
- `eval` 输出 schema 更新（见 Workstream B）

---

## 2. Workstream B：补齐能量评估链路（P3-MVP）
### B1. run_metadata（采样时写入）
在每次采样输出目录写 `run_metadata.json`（或写入统一 metadata 文件），必须包含：

**依赖状态（解释 energy_available_rate）**
- `mlip.name`（如 CHGNet）  
- `mlip.version`  
- `mlip.checkpoint_id/hash`（或模型文件路径）  
- `device` / `dtype`  
- `loaded_ok: true/false`  
- `error_if_failed`（字符串，失败时必须写）

**relax 配置（解释 success_energy 判据）**
- `relax.max_steps`  
- `relax.fmax_threshold`  
- `relax.cell_opt`（是否优化晶胞）  
- `relax.constraints`（如 2D：c 固定/只优化 in-plane 等）  
- `relax.time_limit`（可选）

### B2. eval schema：强制拆分成功标记
在 `twodgen/evaluate/eval_samples.py`（或同等评估入口）输出：
- `success_geom`：几何合法（angle/cond/inplane/collision 等）  
- `energy_available`：mlip 可用且能返回能量/力  
- `success_energy`：relax 成功收敛 + 数值正常（非 NaN / 非爆炸）

并增加失败原因字段（强烈建议）：
- `fail_reason_geom`：`angle_oob | cond_violation | inplane_degen | collision | ...`  
- `fail_reason_energy`：`mlip_missing | load_fail | runtime_error | non_converge | nan_energy | nan_force | ...`

> 目的：避免“能量缺失掩盖几何改进”。即使能量不可用，也必须可诊断并可统计。

### B3. 最小验收
- `energy_available_rate` 不再“无解释地为 0”：  
  - 要么 >0；  
  - 要么 `fail_reason_energy` 全部落在 `mlip_missing/load_fail` 且 metadata 明确写明。  
- `success_geom` 与 `success_energy` 两条曲线可以独立解读（且失败原因统计稳定输出）。

### B4. 交付物
- `schema_samples.md`：记录样本目录结构 + run_metadata 字段说明  
- `eval_report.json`：汇总 `success_geom/success_energy` + 失败原因分布  
- `twodgen/scripts/eval_with_energy.sh`：一键评估（带依赖检查）

---

## 3. Workstream C（可选）：层状材料 MLIP 小规模微调（P4-Optional）
> 前置条件：Workstream B 跑通并能稳定得到 `success_energy` 与 failure taxonomy。

### C1. 数据与任务定义（最小版本）
- 选择一批层状材料结构（以你现有数据/采样输出为主，确保标签/参考 relax 可用）  
- 训练目标（优先级）：force → energy（可选）→ stress（可选）  
- 固定 train/val/test 切分，确保可回归。

### C2. 训练与评估指标
- `force_MAE`（val/test）为主；`energy_MAE` 为辅  
- `relax_success_rate`（val/test）  
- `nan_rate`（能量/力 NaN 的发生率）  
- 运行时间粗统计（推理与 relax）

### C3. Go/No-Go 判据（建议写死）
满足任意两条才进入“替换线上权重”评审：
- `relax_success_rate` 明显上升  
- `force_MAE` 明显下降  
- `success_energy_rate` 上升且 `fail_reason_energy` 中 `non_converge/nan_*` 明显下降  
- 性能/耗时不显著恶化（否则仅作为离线或辅助）

### C4. 交付物
- `mlip_finetune_config.yaml`（或等价配置）  
- `mlip_finetune_report.md`：包含指标表与 go/no-go 结论  
- （可选）`model_registry.json`：记录候选权重与 hash

---

## 4. 最小回归与测试（建议尽早补齐）
### T1. 采样投影回归
- 构造极端 angle/cond/inplane case，验证：  
  - projection on 时，输出统计均为 finite  
  - after 投影，`angle_oob/cond_violation/inplane_degen` 显著下降  
- 断言 `project_trigger_any_rate` 合理（至少能触发）

### T2. 能量链路回归
- 在 mlip 缺失/加载失败时：  
  - `energy_available=false`  
  - `fail_reason_energy` 正确分类  
  - metadata 写明错误  
- 在 mlip 可用时：  
  - `energy_available=true` 且 relax 有明确成功/失败输出

---

## 5. 执行顺序（推荐）
1) **Workstream A（投影兜底）先落地**：实现 + AB 对照，确保 `success_geom_rate` 明显提升。  
2) **Workstream B（能量链路）同步推进**：run_metadata + success_geom/success_energy + 失败原因分类。  
3) 当 A 与 B 都稳定后，再决定是否启用 **Workstream C（MLIP 微调）**。  

---

## 6. 交付物清单（落盘目录建议）
- `twodgen/scripts/`  
  - `debug_cond_trigger.sh`（已有）  
  - `sampling_projection_ab.sh`（新增）  
  - `eval_with_energy.sh`（新增）  
- `twodgen/evaluate/`  
  - schema & failure taxonomy 更新  
- `docs/`（或等价目录）  
  - `schema_samples.md`  
  - `mlip_finetune_report.md`（可选）

---

## 7. Done 定义（最终验收）
- 投影开启后：  
  - `angle_out_of_range_rate`、`cond_lattice_violation_rate`、`inplane_degen_rate` 明显下降  
  - `success_geom_rate` 明显上升  
  - `project_delta_norm_p95` 不至于极端（护栏不应“重写”分布）  
- 能量链路：  
  - `energy_available_rate` 与 `success_energy_rate` 可解释、可统计  
  - failure taxonomy 稳定输出，能定位依赖/配置/数值问题  
- MLIP 微调（若启用）：  
  - 明确 go/no-go 结论，并可复现实验配置与指标表
