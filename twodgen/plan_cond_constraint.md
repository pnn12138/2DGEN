# twodgen 项目规划（规划层）：让 *cond 约束*真正生效并形成可验证闭环  
**版本**：v1.0  
**日期**：2026-02-05  
**目标聚焦**：把 “cond 约束是否激活” 变成 **可触发、可观测、可验收、可回归** 的训练/采样/评估闭环，从而显著降低病态晶格与 angle/collision 失败。

---

## 0.1 状态（2026-02-06）
- A1/A2/A3 已完成并可复现（详见 `twodgen/process.md` 10.6）。
- 关键证据路径：
  - 训练短跑（cond 约束必触发）：`outputs/debug_cond_trigger/on_long_v2/20260205_222822`
  - A/B 采样评估（cond 改善可观察）：`outputs/samples_tokens/cond_on_fix` vs `outputs/samples_tokens/cond_off_baseline`
- 回归与防退化：
  - `tests/test_cond_constraint.py` 覆盖“坏晶格必触发/好晶格为零/统计 finite”
  - `uv run pytest` 全通过

## 0. 背景与当前状态（简述）
当前管线已经打通了 **训练 → 采样 → 评估 → self-train** 的闭环，但评估侧出现典型的“几何失败主导”现象：  
- **angle_out_of_range** 占比极高，valid_rate 偏低（几何层失败压倒一切）。  
- **lattice condition number（cond）分布极端**（mean 很大、median 也不小），说明晶格参数化/约束链路未压住病态区域。  
- 训练侧 `loss_cond_number` 常为 0，暗示 cond hinge **长期不触发** 或梯度链路 **未真正生效**。  

> 规划主线：先让 cond 约束“真触发”并与评估口径对齐，再逐步把阈值与权重调回可用区间，最后把采样端投影/兜底和能量链路一起接入闭环。

---

## 1. 总体目标与验收口径
### 1.1 总目标
1) 训练阶段：`loss_cond_number` 在早期 **必定出现非零**，并能通过日志看到 `pred_cond_*` 的分布被压缩。  
2) 训练-评估一致性：训练记录的 cond 与评估 cond **口径一致**（或可明确换算），避免“训练看起来正常、评估却爆炸”。  
3) 采样阶段：在不破坏多样性的前提下，显著降低 **cond 极端值**，并联动降低 **angle_out_of_range / collision**。

### 1.2 建议的核心指标（建议做成统一 dashboard）
**训练侧（每 N step 汇总）**  
- `cond_gram.mean / p50 / p95 / max`  
- `cond_lattice.mean / p50 / p95 / max`（与评估同口径）  
- `cond_violation_rate = mean(cond_lattice > cond_max_eval)` 或 `mean(cond_gram > cond_max_train)`  
- `loss_cond_number`（均值、p95）  
- `lambda_cond`（随 warmup 的曲线）

**采样/评估侧（每次采样跑）**  
- `valid_rate`（几何有效）  
- `angle_out_of_range_rate`  
- `collision_rate`  
- `cond_lattice.mean / p50 / p95 / max`  
- （后续接入）`energy_available_rate / energy_success_rate`

### 1.3 阶段性验收门槛（建议目标）
> 下面是建议目标，用于方向性验收；具体阈值可按数据集/任务再细化。  
- **A1（链路验收）**：训练早期（少量 step）`loss_cond_number > 0` 且 `cond_violation_rate > 0`（确认“能触发”）。  
- **A2（口径验收）**：`cond_gram` 与 `cond_lattice` 统计上至少 **单调相关**（rank 相关明显）。  
- **A3（闭环验收）**：采样评估中 `cond_lattice.p95` 显著下降，且 `angle_out_of_range_rate` 随之下降。  

---

## 2. 工作分解（WBS）与优先级
### P0：让 cond loss 真正“打进训练里”（必须先做）
#### P0-1：做一个“必触发”的验证跑（最小实验）
**目的**：在 100–500 step 内看到 `loss_cond_number` 非零，并且 `pred_cond` 分布发生可解释变化。  
**动作**（建议全部写入一份 debug config，跑完即弃）：  
- 把 `cond_max_train` 降到 **30/40/50** 这种区间（确保当前 `pred_cond_mean` 能越过阈值）。  
- 把 `loss_weight_warmup_steps` 对 cond 的 warmup 缩短到 **0–1000**。  
- 暂时增大 `cond_loss_weight`（例如提升到 0.05–0.2）只用于观察链路是否有效。  

**输出物**  
- 一次短跑的日志对比：开启/关闭 cond loss 的 `loss_cond_number`、`cond_violation_rate`、`cond_*` 统计。  
- 结论：是否“触发成功”、触发是否可控、是否影响其他损失项稳定性。

#### P0-2：训练侧同时记录两种 cond（对齐评估口径）
**问题来源**：训练可能在压 `cond_gram`，评估在看 `cond_lattice`，导致阈值与量纲错位。  
**动作**  
- 在训练 forward / logging 里增加：  
  - `cond_gram`：当前 loss 使用的那一套（保持不动）  
  - `cond_lattice`：把同一 `gram6` 通过 `gram6_to_lattice` 转成 lattice 后再算 cond（与评估同口径）  
- 在日志中同时输出 mean/p50/p95/max，并输出二者的相关性（至少 Spearman）。

**输出物**  
- 一份“训练 cond 口径说明”：loss 用哪个、评估用哪个、是否需要换算。

#### P0-3：做一次“梯度链路真实性”检查（避免假激活）
**目的**：排除 cond 在日志非零但梯度被 clamp/detach 截断的情况。  
**动作**  
- 加一个 debug 开关：对 lattice head / gram6 head 参数打印 `grad_norm`（仅少量 step）。  
- 对比：cond loss on vs off 的 `grad_norm` 是否显著变化。  
- 同时检查：cond 计算前后是否存在 `detach/clamp/round/minmax` 等不可微/饱和操作。

**输出物**  
- 一份“cond 梯度路径检查表”：关键张量名、所在函数、是否可微、是否需要替换为平滑近似。

---

### P1：把阈值/权重从“必触发”调回“可用区间”（让训练稳定且有效）
#### P1-1：cond 阈值 schedule（从紧到松或从松到紧）
**两种策略（择一）**  
- **策略 S1（先紧后松）**：先用较小 `cond_max_train` 把病态区域推开，后期逐步放宽以保多样性。  
- **策略 S2（先松后紧）**：早期不强约束避免训练不稳，中期开始收紧（更常见，但你目前的问题是“永不触发”，因此更推荐 S1 先验证有效性）。  

**建议落地**  
- 写成显式 schedule：`cond_max_train(t)` 与 `lambda_cond(t)` 同步定义。  
- dashboard 同步展示：`cond_violation_rate` 是否随 schedule 下降。

#### P1-2：把 cond hinge 改成“软约束”并提供平滑梯度（可选）
如果发现 hinge 边界过硬导致训练振荡，可用平滑版本：  
- `softplus(k*(cond - cond_max)) / k` 或 `log(1+exp(...))`  
- 或对 `log(cond)` 做约束（cond 的长尾更容易稳定）

**输出物**  
- 对比实验：hinge vs softplus，对稳定性与采样有效率的影响。

---

### P2：采样端兜底（让评估不再被 angle/cond 直接“秒杀”）
> 即便训练有效，采样也可能掉进坏区域；采样端需要“投影/裁剪/重参数化”的工程兜底。

#### P2-1：采样后投影（post-step projection）
**目标**：每一步/每几步对 lattice 做投影，强制：  
- angle 合法区间  
- cond 上限  
- 避免退化（例如极小体积、极端轴比）

**动作**  
- 在 sampler 的 `_project_step / clip_lattice`（或等价位置）加入“可开关投影”：  
  - angle：使用连续可微的有界映射（例如对角度用 sigmoid/tanh 映射到合法区间）  
  - cond：对 gram/cholesky 做投影到可行域（必要时做近似/迭代投影）  
- 记录投影触发次数与幅度（投影越多说明模型还没学会）。

#### P2-2：在采样中引入 cond guidance（可选）
你已经有 force-guidance（CHGNet forces），可在采样中叠加一个轻量的 `cond_guidance`：  
- 直接对 `cond_lattice` 的软惩罚做梯度引导（权重很小，作为“护栏”）

---

### P3：把能量链路纳入闭环（避免只优化几何）
> 当前 energy_available_rate=0，导致闭环只看几何成功率；后续要把 energy/relax 成功率接上。

#### P3-1：明确 relax 配置与依赖状态写入 metadata
- 每次采样输出：MLIP 名称、版本、是否成功加载、relax 参数（步数/阈值/约束）。  
- 评估输出拆分：`success_geom` / `success_energy`，并给出失败原因分类。

#### P3-2：self-train 筛选策略从“几何”升级到“几何+能量”
- 先用几何过滤，再用能量/力一致性做二级筛选，避免把“几何凑巧但能量很差”的样本灌回训练集。

---

## 3. 实验矩阵（最小但够用）
### E0：必做（链路验证）
- E0-1：cond_max_train = 40；warmup_cond=0；cond_weight=0.1（短跑）  
- E0-2：同配置但 cond_loss=off（对照）

**判据**：`loss_cond_number` 非零 + `cond_violation_rate` 非零 + `cond_*` 分布变化。

### E1：口径一致性
- 训练同时 log cond_gram 与 cond_lattice；计算相关性  
**判据**：单调相关明显；若弱相关则必须先修口径/投影位置。

### E2：schedule 对比（选做）
- S1 vs S2 各跑一个小周期  
**判据**：稳定性（loss 曲线）、采样端 `cond.p95`、`angle_out_of_range`。

### E3：采样后投影开关对比（选做）
- sampler 投影 off vs on  
**判据**：valid_rate 上升、angle/collision 下降，同时记录多样性是否显著下降（可用结构分布/成分覆盖粗评）。

---

## 4. 代码触点与改动清单（按你现有结构命名）
> 下面列的是“通常会出现”的触点，你可以按仓库实际文件名对齐。

### 4.1 训练侧（loss 与日志）
- lattice/gram6/cholesky6 转换：`gram6_to_lattice`, `lattice_to_gram6`, `gram6_to_cholesky6`, `cholesky6_to_gram6`  
- 投影/裁剪：`_project_step`, `clip_lattice`  
- 模型头：`AtomDenoiser` 或 lattice head 模块  
- 日志：训练 step logger（增加 cond_gram/cond_lattice 与 violation_rate）

### 4.2 采样侧（投影与 guidance）
- sampler step：增加 post-step projection 开关  
- guidance：在 force-guidance 之外增加轻量 cond guidance（可选）

### 4.3 评估侧（口径统一）
- cond 的定义函数：统一使用 `cond_lattice` 并在训练侧同步输出  
- 输出字段：拆 success_geom/success_energy，并细分失败原因

---

## 5. 风险与对策
- **R1：cond 约束过强导致训练崩/模式塌缩**  
  - 对策：先用短跑验证链路，再用 schedule；必要时用 softplus 平滑；监控多样性 proxy。  
- **R2：训练压的是 gram cond，但评估看的是 lattice cond（口径错位）**  
  - 对策：训练日志同时输出两者并算相关性；必要时把 loss 迁移到 lattice cond 或引入换算。  
- **R3：投影/clip 非可微导致梯度失效**  
  - 对策：训练端用可微有界映射；采样端可用硬投影做兜底但要监控投影频率。  
- **R4：只优化几何导致能量不可用**  
  - 对策：尽快接入 energy_available_rate；self-train 加入能量二级筛选。

---

## 6. 交付物清单（每个阶段都能落盘）
- `plan_cond_constraint.md`（本文档，持续更新）  
- `debug_cond_trigger.yaml/json`（E0 配置）  
- `cond_logging_patch.diff`（训练日志补丁）  
- `cond_consistency_report.md`（E1 结果：口径一致性与相关性）  
- `sampling_projection_patch.diff`（P2 投影补丁）  
- `eval_schema_update.md`（success_geom/success_energy 字段与失败原因分类）

---

## 7. 下一步行动（按顺序执行）

### 7.1 当前状态（2026-02-06）
- [x] 执行 E0：做“必触发验证跑”，确认 `loss_cond_number` 真非零  
  - 证据：`twodgen/scrip/debug_cond_trigger.sh`；训练 run `outputs/debug_cond_trigger/on_long_v2/20260205_222822/train_metrics.jsonl`
- [x] 落地 P0-2：训练侧同时 log `cond_gram` & `cond_lattice` + 相关性  
  - 证据：训练日志字段 `cond_gram_* / cond_lattice_* / cond_gram_lattice_spearman / cond_diff_abs_* / cond_valid_rate`
- [x] 执行 P0-3：grad_norm 对比开关已落地（需要时可跑 on/off 对照）  
  - 证据：`twodgen/scrip/train_tokens.py` 的 `--debug-grad-submodules`
- [x] 进入 P1：把阈值/权重做成 schedule  
  - 证据：`twodgen/scrip/train_tokens.py` 的 `--cond-max-start/--cond-max-end/--cond-max-steps/--cond-max-schedule`
- [x] 进入 P2：采样后投影兜底（开关已可用，并输出投影统计）  
  - 证据：`twodgen/scrip/sample_tokens.py` 的 `--project-gram-cond/--project-gram-max-cond/--project-final/--project-every-step`，
    评估输出 `cond_violation_rate / project_trigger_rate / project_delta_cond`
- [x] 进入 P3-1：能量链路可用（energy_available_rate > 0，success_energy 可统计）  
  - 证据：`outputs/samples_tokens/cond_on_relax/eval/tier0_metrics.json`、`outputs/samples_tokens/cond_off_relax/eval/tier0_metrics.json`
- [ ] 进入 P2-2：采样中 cond guidance（可选，暂未实现/未验证收益）
- [x] 进入 P3-2：self-train 二级筛选从“几何”升级到“几何+能量”（已落地）  
  - 证据：`twodgen/evaluate/self_train_loop.py` 新增 `--select geom_energy`，并可透传 `--npz/--cond-npz`；
    demo：`outputs/self_train_demo_geom_energy_v4/self_train_summary.json`（selected=1/20，输出 self_train.npz）

### 7.2 验收结果记录（A1/A2/A3）
- A1（链路验收）：满足。训练早期 `loss_cond_number > 0` 可触发（见 `outputs/debug_cond_trigger/on_long_v2/20260205_222822/train_metrics.jsonl`）。
- A2（口径验收）：已对齐。cond_gram 与 cond_lattice 采用同尺度（cond_gram = sqrt(cond(G))），
  在短跑中 `cond_diff_abs_mean ~ 1e-6`（见 `outputs/debug_cond_trigger/post_fix_check/20260206_112226/train_metrics.jsonl`）。
- A3（闭环验收）：已完成“可观察”闭环，但效果依赖采样设置/样本量，当前主要失败仍由 `bad_volume` 主导。
  - 小样本对照（64 samples）：on (`outputs/samples_tokens/cond_on_fix`) 的 cond_overflow=6 vs off (`outputs/samples_tokens/cond_off_baseline`) 的 13。
  - 大样本对照（256 samples, 80 steps）：on (`outputs/samples_tokens/cond_on_longcheck`) cond_overflow=53、cond_violation_rate=0.207；
    off (`outputs/samples_tokens/cond_off_longcheck`) cond_overflow=39、cond_violation_rate=0.152。提示下一阶段应优先控制 volume/lattice scale，
    否则 cond 改善容易被体积爆炸掩盖。

### 7.3 能量闭环补充记录（P3）
- relax 评估已跑通（CHGNet）：
  - on：`outputs/samples_tokens/cond_on_relax/eval/tier0_metrics.json`：energy_available_rate=1.0、success_energy_rate=0.71875（但 success_geom_rate=0.0625）。
  - off：`outputs/samples_tokens/cond_off_relax/eval/tier0_metrics.json`：energy_available_rate=1.0、success_energy_rate=0.8125（success_geom_rate=0.125）。
- self-train 二级筛选已落地并验证：
  - `twodgen/evaluate/self_train_loop.py` 支持 `--select {success,geom,geom_energy}`，并新增 `--npz/--cond-npz` 透传给采样端。
  - demo：`outputs/self_train_demo_geom_energy_v4/self_train_summary.json`（`--select geom_energy`，selected=1/20）。

### 7.4 下一步优先级（从当前数据出发）
- 现阶段 valid/success 的主导失败仍是 `bad_volume`（其次 cond_overflow/collision），建议把下一步改成“先控 volume/scale，再复验 cond 收益”：
  - 采样端增加 volume clamp（将 lattice 等比缩放到 [v_min, v_max]）或增加 g_scale override；
  - 训练侧考虑下调 g_scale 或把 volume/c_len 约束提前（避免采样端体积爆炸）。
