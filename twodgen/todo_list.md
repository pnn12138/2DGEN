# 待做任务清单（对齐 `twodgen/milps-plan.md`：v1.0 / 2026-01-29）

> 说明：本清单以 milps-plan 的 Phase 1/2/3 为主线；**优先用“Fixer（MLIP + 几何约束）”把有效率拉到可用**，再做训练侧约束与架构升级。已完成条目已从本清单移除。

## 状态
- todo
- review
- done

## phase1（目标2：修正晶格角度与条件数畸变）
### 1.1 todo（代码级实施规划，按落地顺序）
（已全部完成，见 1.3）

### 1.2 review
- 对照 1.1 的输出逐条验收：日志/指标是否出现、统计是否改善、是否影响采样稳定性与多样性。

### 1.3 done
- 1.1.1 收敛“修复入口”到唯一 lattice 表示（权威表示）
  - 权威表示定义为 `gram6`，投影/损失统一走 Gram 表示；`_project_step` 只对 Gram 做投影。
  - 输出“权威表示规范表”（见 `twodgen/process.md` 0.1）。
- 1.1.2 梯度路径表 + 最小 autograd 断言（防断梯度）
  - 新增 `tests/test_crystal.py` 里的 `test_gram6_roundtrip_has_gradients` 覆盖 SPD→Gram→lattice→Gram 的梯度回传。
- 1.1.3 cond 硬投影（SPD 特征值投影）
  - 新增 `project_gram_cond_spd`（log/linear 模式），采样 CLI 支持 `--project-gram-cond/--project-gram-max-cond` 并写出 cond 前后统计。
- 1.1.4 角度有界映射（优先输出 cos，再 arccos）
  - 角度计算支持 `angle_param_mode=raw|cos|sigmoid`，CLI 已接入。
- 1.1.5 训练侧惩罚改为“分段光滑 barrier”
  - angle/cond 换成 softplus barrier，cond 走 log 空间。
- 1.1.6 采样端投影位置与模式
  - 支持 `--project-every-step/--project-final`，采样导出每步投影统计均值。
- 1.1.7 合法性检查 + 标准化 fail_reason
  - 标准化 `angle_out_of_range/non_spd/cond_overflow/det_nonpos`，采样/评估输出 Top3 统计。
- 1.1.8 统计工具与回归验证
  - `eval_samples.py` 增加 cond_violation/project_trigger/delta_cond 统计；新增投影与梯度回归测试。

---

## phase1b（目标2-补强：让 cond 约束“可触发且可验证”）
> 对齐 `twodgen/process.md` 10.3/10.4 的现状结论，并落到 `twodgen/plan_cond_constraint.md`（v1.0 / 2026-02-05）的代码级 WBS。

### 2.1 done
- 2.1.1 E0 on/off 短跑脚本 `twodgen/scrip/debug_cond_trigger.sh`，固定关键参数，验收口径写入脚本注释。
- 2.1.2 训练日志新增 cond_p50/p95/max、violation_rate、valid_rate，JSON 同步。
- 2.1.3 双口径 cond_gram / cond_lattice + abs/rel diff + Spearman，对齐评估口径。
- 2.1.4 `--debug-grad-submodules` 梯度探针，log_interval 输出子模块 grad_norm。
- 2.1.5 cond_max schedule（linear/cosine）CLI 接入，训练循环实时更新。
- 2.1.6 回归测试 `tests/test_cond_constraint.py` 覆盖坏晶格触发、好晶格为零、统计均有限。

### 2.2 review
- 对照 `twodgen/plan_cond_constraint.md` 的 A1/A2/A3：
  - A1：短跑必触发（loss/violation 非零）
  - A2：cond_gram vs cond_lattice 单调相关/差值极小
  - A3：采样评估侧 `cond.p95` 与 angle/collision 失败率联动下降

### 2.3 done
- 2026-02-06：A1/A2/A3 验证完成。训练侧 cond_gram 取平方根与 cond_lattice 对齐（差值均值 <5e-6），短跑 on 组 `loss_cond_number` 早期触发（run=outputs/debug_cond_trigger/on_long_v2/20260205_222822）。采样评估对照：on 组 (`outputs/samples_tokens/cond_on_fix`) cond_violation_rate 0.094 vs off 组 (`outputs/samples_tokens/cond_off_baseline`) 0.203，且 cond_overflow 失败数从 13 → 6；角度/碰撞失败率持平，cond 改善已可观察。
- （完成后把关键结论同步回写到 `twodgen/process.md` 的“下一步动作/实验矩阵”段落，并在 `twodgen/history.md` 追加条目化记录）

---

## phase2（目标3：采样端硬约束投影兜底 + 能量评估链路补齐）
> 对齐 `twodgen/plan_next_sampling_energy_mlip.md`（v1.0 / 2026-02-06）。本 phase 聚焦：P2-MVP（采样端 post-step 投影）+ P3-MVP（能量链路 run_metadata + failure taxonomy）；
> 可选：P4（MLIP 微调）先只做“接口与 go/no-go 产物”，不强行推进训练。

### 3.1 todo（代码级实施规划，按落地顺序）
- 3.1.1 采样端 post-step 投影入口（MVP：angle/cond/inplane/volume）
  - 新增模块 `twodgen/common/projection.py`（或同等位置）：
    - **坐标一致性（必须写清楚）**：
      - `coords_mode: "frac"|"cart"`：若为 cart，需同步 `cart -> frac(old) -> cart(new)`，避免投影后几何失真
    - **cell 参数域投影（推荐做法，避免直接改 lattice）**：
      - `lattice_to_cell(lattice)` / `cell_to_lattice(cell)`（含 handedness fix：`det>0`，以及 a/b/c 下界）
      - `project_angles_cell(cell, angle_min, angle_max, mode="clamp")`
      - `project_inplane_cell(cell, *, a_min, b_min, gamma_min, gamma_max, area_min)`（优先只修 in-plane）
    - **cond 投影（与 angle/inplane 互相打架要处理）**：
      - `project_cond_svd(lattice, cond_max)`：SVD 裁剪 `sigma_min >= sigma_max/cond_max`
      - 顺序建议：`angle -> inplane -> cond -> (再做一次 angle/inplane clamp)`；或小循环（最多 2 次）直到满足或到上限
      - 明确 cond 口径：与 `eval_samples.py` 一致；2D slab（pbc_mask=1,1,0）默认用 **in-plane Gram cond**（避免真空轴主导 cond）
    - **volume 兜底（避免 bad_volume 统治失败原因）**：
      - `project_volume_inplane(lattice, v_min, v_max, pbc_mask)`：只缩放/放大 in-plane 两个 lattice 向量，使 volume 回到训练数据 p1/p99 区间（不动真空轴）
      - 约束：缩放下界必须保证 `a/b/area` 不掉到 `inplane_*_min` 以下（否则会立刻变成 inplane_degenerate）
    - 返回 `(lattice_new, stats_dict)`：必须包含 before/after 的 angle_oob/cond/area/inplane_degen 与 `delta_norm`
  - `twodgen/scrip/sample_tokens.py`：
    - 新增 CLI：`--post-project --post-project-interval {0,1,5,10} --post-project-keys angle,cond,inplane,volume --post-project-v-min/--post-project-v-max`
    - 在 sampler 主循环“更新 lattice 后”插入 `post_step_project()`（按 interval 与 keys 控制）
    - 投影统计落盘建议：优先写 `run_metadata.json` 或 `projection_stats.json`（run-level 汇总），eval 侧合并写入 tier0/tier1（避免塞进 samples.npz 导致膨胀/口径漂移）

- 3.1.2 评估侧增加 in-plane 退化判定 + 几何失败原因拆分
  - 修改 `twodgen/evaluate/eval_samples.py`：
    - 增加 `inplane_degen` 判定（a/b/gamma/area），并输出 `inplane_degen_rate`
    - `fail_reason_geom` 细分：`angle_oob | cond_violation | inplane_degen | collision | bad_volume | ...`
      - **主失败原因优先级（必须写死）**：建议 `collision > angle_oob > inplane_degen > cond_violation > bad_volume > other`
      - （可选）保留 `fail_reasons_geom: list[str]` 作为附加诊断，但 tier0 用主原因
    - tier0/tier1 增加：`cond_lattice_violation_rate`、`cond_lattice_p95/max`、`project_trigger_any_rate`、`project_delta_norm_p95`
  - 验收口径：projection on 时 angle/cond/inplane 相关 fail_reason 显著下降，且 fail_reason 发生“可解释迁移”

- 3.1.3 能量链路 run_metadata（采样时写入）+ energy failure taxonomy
  - 修改 `twodgen/scrip/sample_tokens.py`：
    - 在输出目录新增 `run_metadata.json`（或将 sampling_config.json 扩展为等价字段）：
      - `mlip.name/version/loaded_ok/error_if_failed/device/dtype`
      - `relax.max_steps/fmax/cell_opt/constraints/...`
      - 复现增强（建议）：`git_commit`、`ase/pymatgen/chgnet` 版本号
  - 修改 `twodgen/evaluate/eval_samples.py`：
    - 输出 `energy_available`、`success_energy`、`fail_reason_energy`
      - `mlip_missing | load_fail | runtime_error | non_converge | nan_energy | nan_force | ...`
    - **必须区分“没跑” vs “跑了失败”**：
      - 增加 `energy_skipped_reason = "geom_fail"|"mlip_unavailable"|None`
      - 逻辑建议：仅 `success_geom==True` 才尝试 relax；否则能量标记为 skipped（避免算力浪费 + 统计一致）
    - 保证 energy_available_rate=0 时可从 metadata 与 fail_reason_energy 解释（不再黑箱）

- 3.1.4 A/B 一键脚本（回归最小实验矩阵）
  - 新增 `twodgen/scrip/sampling_projection_ab.sh`：
    - A：projection off
    - B：`--post-project --project-interval 1 --project-keys angle,cond,inplane`
    - 固定 seed、num_samples、steps、cond_max_eval 等，输出到同一父目录方便 diff
  - 新增 `twodgen/scrip/eval_with_energy.sh`：检查 chgnet/ase 依赖，跑 sampling + eval，确保 energy_available 可解释

- 3.1.5 最小回归测试
  - 新增 `tests/test_sampling_projection.py`：
    - 构造 angle_oob/cond_overflow/inplane_degen lattice，断言 projection 后指标回到阈值内且统计 finite
  - 新增 `tests/test_energy_chain.py`：
    - 模拟 mlip 缺失/加载失败：断言 `energy_available=false`、`fail_reason_energy` 分类正确、metadata 写 error
    - mlip 可用 smoke（可选，标 `pytest.mark.slow`）：`--relax` 跑 1 个结构确保能量字段非空

- 3.1.6 可选：采样 cond guidance（P2-2）
  - `twodgen/scrip/sample_tokens.py` 新增 `--cond-guidance/--cond-guidance-scale/--cond-guidance-start`
  - 验收：在 projection 触发率不显著上升的前提下，cond.p95/cond_violation 下降

- 3.1.7 可选：MLIP 微调（P4-Optional，仅产物与 go/no-go）
  - 新增文档 `twodgen/mlip_finetune_report.md`（或写入 process 段落）定义数据/指标与 go/no-go
  - 若落地训练脚本：优先放在 `P_TASK/` 或 `twodgen/scrip/`（避免新增顶层目录）
  - 2026-02-06：已新增模板与登记文件（未训练）：`twodgen/mlip_finetune_report.md`、`twodgen/mlip_finetune_config.yaml`、`twodgen/model_registry.json`

### 3.2 review
- A/B 对照：projection on 必须使 `angle_out_of_range_rate/cond_lattice_violation_rate/inplane_degen_rate` 明显下降，并抬升 `success_geom_rate`
- 能量链路：`energy_available_rate` 与 `success_energy_rate` 可解释且 failure taxonomy 稳定输出
- 投影副作用：`project_delta_norm_p95` 不应极端；若 collision 上升，调整投影顺序/幅度
  - 2026-02-06：新增 volume clamp 后 A/B 已满足“success_geom 显著上升”的验收（见 `outputs/ab_proj_phase2_v8`：A success_geom_rate=0.133，B=0.484；bad_volume 98→47；post_project_trigger_any_rate=0.742；collision 19→20）。

### 3.3 done
- 2026-02-06：phase2 MVP 已落实（对应 `twodgen/plan_next_sampling_energy_mlip.md` Workstream A/B/C/D/E）：
  - A（采样投影兜底）：新增 `twodgen/common/projection.py`；`twodgen/model/atom_denoiser.py` 支持 `--post-project/--post-project-interval/--post-project-keys`，并导出 per-sample & per-step 统计；`twodgen/scrip/sample_tokens.py` 写 `run_metadata.json` 与 `projection_stats.json`。
    - 2D cond 口径修正：cond clamp 与 eval 统一为 **in-plane Gram cond**（pbc_mask=1,1,0）。
    - 新增 `volume` 投影 key：默认从训练 npz 自动读取 volume bounds（p1/p99），只缩放 in-plane 以压制 bad_volume。
  - B（评估失败原因/统计）：`twodgen/evaluate/eval_samples.py` 增加 `inplane_degenerate` 判定与 `fail_reason_geom` 主原因（固定优先级），输出 `cond_lattice/inplane_*` 与 post-project 统计字段。
  - C（能量链路 taxonomy）：`eval_samples.py` 增加 `energy_skipped_reason/fail_reason_energy` 与跳过率统计（geom_fail vs mlip_unavailable），energy_available 由 energy_mlip 驱动而非依赖 element refs。
    - relax gate：采样端仅对几何成功（valid & non-cross-vacuum）样本尝试 CHGNet relax，避免算力浪费并使 energy_skipped_reason 更一致。
  - D（A/B 脚本）：新增 `twodgen/scrip/sampling_projection_ab.sh`、`twodgen/scrip/eval_with_energy.sh`。
  - E（最小测试）：新增 `tests/test_sampling_projection.py`、`tests/test_energy_chain.py`，并修复旧测试 `tests/test_c2db_clean_2d.py` 的 import；`uv run pytest` 全通过。
- 待做（可选）：3.1.6 cond guidance、3.1.7 MLIP 微调 go/no-go（保持可选即可）。
