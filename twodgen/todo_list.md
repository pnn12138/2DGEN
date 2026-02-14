# 待做任务清单（对齐 `twodgen/plan.md`：v0.1 / 2026-02-10）

> 说明：本清单仅保留“新一轮未完成项”，已完成历史条目已移除。目标是把 E0–E5 做成可复现实验流水线与论文产物。

## 状态
- todo
- review
- done

## 当前进度快照（2026-02-10，已更新）
- `phase0`：done
  - E0 协议链路已通过（schema/STATUS/resume/validator）。
- `phase1`：review
  - `E1_1` quick 结果已产出：`delta_success_geom_rate=+0.1133`（未达 `+0.15` 阈值）。
  - `E1_2`/`E1_3` 已有脚本与配置，当前进入“待复跑 + 待验收”状态。
- `phase2`：review
  - 训练 proxy 指标、E2 对照脚本、trigger trend 检查工具已落地，待真实训练 run 验证趋势是否下降。
- `phase3`：review
  - soft/hard symmetry 配置与评估枚举已落地，待 final 口径阈值验收（`spacegroup_match_rate`/`spglib_fail_rate`）。
- `phase4`：review
  - screening/DFT 导出回填链路已落地，待真实 DFT 回填验证。
- `phase5`：review
  - novelty/diversity/QD/mode-collapse 工具已落地，待 coverage/novelty 阈值验收。
- `phase6`：review
  - paper assets 导出与 repro manifest 已落地，待一键重建验收。

---

## phase0（实验协议固化与产物规范）
### 0.1 todo（代码级实施规划，按落地顺序）
- 0.1.0 核心 JSON Schema + 版本化（P0）
  - 新增 `twodgen/evaluate/schemas/`：
    - `run_metadata.schema.json`
    - `metrics_summary.schema.json`
    - `failure_breakdown.schema.json`
    - `projection_stats.schema.json`
  - 以上 JSON 顶层强制字段：
    - `schema_version`
    - `git_commit`
    - `timestamp`
    - `experiment_id`
    - `config_hash`
    - `seed`
    - `protocol`
  - `twodgen/evaluate/io.py` 改为按 `schema_version` 分支兼容读取，禁止“猜字段”。
- 0.1.1 统一 run 目录与产物命名
  - 新增 `twodgen/evaluate/run_layout.py`（或同等模块），统一创建：
    - `runs/<EXPERIMENT_ID>/<YYYYMMDD_HHMMSS>/`
    - `run_metadata.json`、`projection_stats.json`、`metrics_summary.json`、`failure_breakdown.json`、`plots/`、`samples/`
  - `twodgen/scrip/sample_tokens.py`、`twodgen/evaluate/run_pipeline.py` 迁移到该布局，保留旧路径兼容读取。
- 0.1.1b 原子写入与失败恢复（P1）
  - JSON 产物采用 `*.tmp -> rename` 原子写入，避免中断产生半文件。
  - 每个 run 写 `STATUS.json`：
    - `running | success | failed`
    - `error_trace.txt`（失败时）
  - runner 支持跳过 success、仅重跑 failed。
- 0.1.2 固化 seeds 与样本预算入口
  - 新增 `twodgen/evaluate/protocol.py`：
    - `quick`: `num_samples=2000`, `seeds=[0,1,2]`
    - `final`: `num_samples=20000`, `seeds=[0,1,2]`
  - 脚本侧禁用“隐式默认 seed”，必须显式写入 metadata。
- 0.1.2b 实验注册表（P0）
  - 新增 `twodgen/configs/bench/experiments.yaml`：
    - 注册 `E1_1/E1_2/E1_3/E2_1/E3_1/E4_1/E5_1` 对应 config、默认 protocol、论文图表归属。
  - `ablation_runner.py`、`export_paper_assets.py` 从 registry 自动发现实验，不再硬编码。
- 0.1.3 指标摘要与失败摘要标准化输出
  - 在 `twodgen/evaluate/eval_samples.py` 增加 `metrics_summary.json` 与 `failure_breakdown.json` 导出函数；
  - `twodgen/evaluate/io.py` 增加读取新文件的兼容逻辑。
- 0.1.4 聚合口径冻结（P0）
  - 新增 `twodgen/evaluate/aggregate_runs.py`（全局统一聚合器）：
    - 输入：`runs/<EXP>/*/metrics_summary.json`
    - 输出：`runs/<EXP>/_aggregate/summary.csv|summary.json|plots/`
  - 口径写死并入 metadata：
    - 先按 seed 聚合，再报告 `mean±std`
    - 默认给出 `95% CI`（样本足够时）
    - 显著性检验（可选）统一为同一检验方法。
- 0.1.5 路径与命名规范化（P0）
  - 新路径统一为 `twodgen/scripts/`；
  - 旧 `twodgen/scrip/` 保留 shim 转发并打印 deprecation 警告，后续窗口再移除。

### 0.2 review
- 跑 E0（N=200, seed=0）确认所有标准产物都生成且字段完整。
- E0 额外验收：
  - schema 校验通过；
  - `STATUS.json` 状态正确；
  - 中断后可续跑且不会污染聚合结果。

### 0.3 done
- 已落地模块：
  - `twodgen/evaluate/run_layout.py`
  - `twodgen/evaluate/protocol.py`
  - `twodgen/evaluate/aggregate_runs.py`
  - `twodgen/evaluate/validate_artifacts.py`
  - `twodgen/evaluate/run_e0.py`
  - `twodgen/evaluate/schemas/*.schema.json`
- 已落地输出：
  - `eval_samples.py` 生成 `metrics_summary.json` 与 `failure_breakdown.json`
  - `sample_tokens.py` 输出 `run_metadata.json` 与 `projection_stats.json` 使用 schema envelope + 原子写入
- E0 实跑通过：`runs/E0/20260210_e0_seed0_n200`（validator 通过，resume 跳过正常）。

---

## phase1（E1：有效性 ablation）
### 1.1 todo（代码级实施规划，按落地顺序）
- 1.1.1 基线 vs 全投影 A/B 统一脚本
  - 新增 `twodgen/scripts/exp_e1_baseline_vs_projection.sh`（A=projection off, B=full projection）。
  - 输出路径必须走 `runs/E1_1/...`，并自动聚合 3 seeds mean±std。
- 1.1.2 投影组件消融矩阵
  - 新增配置组 `twodgen/configs/bench/E1_2_*.yaml`：
    - `cond_only`、`angle_only`、`volume_only`、`cond_angle`、`full`
  - 新增驱动脚本 `twodgen/evaluate/ablation_runner.py`，批量执行并汇总表格。
- 1.1.3 g_scale 扫描
  - 新增 `twodgen/configs/bench/E1_3_gscale.yaml` 与 sweep 脚本；
  - 强制输出 validity-diversity 对照图数据（给 Fig.6 复用）。
- 1.1.4 失败分类与触发率对齐
  - `eval_samples.py` 固化指标：
    - `success_geom_rate`
    - `collision_rate`
    - `cross_vacuum_risk_rate`
    - `inplane_degen_rate`
    - `bad_volume_rate`
    - `post_project_trigger_any_rate`

### 1.2 review
- 验收阈值：`success_geom_rate(B)-success_geom_rate(A) >= 0.15`；
- 同时检查 `bad_volume_rate` 显著下降且 diversity 不塌陷。
- “不塌陷”量化判据（冻结）：
  - coverage 相对 baseline 下降不得超过 20%；
  - novelty 中位数下降不得超过 10%。

### 1.3 done（部分）
- 已落地脚本/模块：
  - `twodgen/evaluate/ablation_runner.py`
  - `twodgen/scripts/exp_e1_baseline_vs_projection.sh`
  - `twodgen/scripts/exp_e1_component_ablation.sh`
  - `twodgen/scripts/exp_e1_gscale_sweep.sh`
  - `twodgen/evaluate/collect_gscale_sweep.py`
- `E1_1` quick 已完成：
  - baseline `success_geom_rate=0.3127±0.0105`
  - full_projection `success_geom_rate=0.4260±0.0044`
  - delta `+0.1133`（未达 `+0.15` 阈值）
- `sample_tokens.py` 已支持 `--g-scale` + `--override-g-scale`（用于 E1.3 sweep）。

### 1.4 review（进行中）
- `E1_2`：运行中（组件消融，`runs/E1_2`）。
- `E1_3`：排队中（`E1_2` 完成后自动执行）。

---

## phase2（E2：训练与采样协同）
### 2.1 todo（代码级实施规划，按落地顺序）
- 2.1.1 约束学习曲线导出
  - 在 `twodgen/scripts/train_tokens.py`（或兼容入口）增加导出字段：
    - `post_project_trigger_rate_train_proxy`
    - `cond_violation_rate_train_proxy`
    - `vacuum_violation_rate_train_proxy`
  - 输出到 `train_metrics.jsonl` + 聚合 json。
- 2.1.2 schedule/repulsion 对照实验脚本
  - 新增 `twodgen/scripts/exp_e2_curriculum_repulsion.sh`，覆盖：
    - 不同 ramp schedule（linear/sigmoid/cosine）
    - repulsion on/off
  - 结果汇总到 `runs/E2_1/.../metrics_summary.json`。
- 2.1.3 “触发率下降”自动验收
  - 新增 `twodgen/evaluate/check_trigger_trend.py`，检查后半程触发率低于前半程并生成判定报告。

### 2.2 review
- 验收口径：在几何成功率不退化前提下，`post_project_trigger_any_rate` 下降。

---

## phase3（E3：对称性可控生成）
### 3.1 todo（代码级实施规划，按落地顺序）
- 3.1.1 soft vs hard symmetry 实验入口
  - 新增配置 `twodgen/configs/bench/E3_1_soft.yaml`、`twodgen/configs/bench/E3_1_hard.yaml`；
  - 保证 `sample_tokens.py` 与 `train_tokens.py` 均可读取同一 symmetry 开关集合。
- 3.1.2 对称性指标补齐
  - `eval_samples.py` 输出并汇总：
    - `spacegroup_match_rate`
    - `spglib_fail_rate`
    - `symmetry_violation_breakdown`
  - 与 `failure_breakdown.json` 对齐字段名，避免重复口径。
- 3.1.2b 对称性失败类型枚举冻结（P2）
  - `symmetry_violation_breakdown` 固定枚举：
    - `spglib_fail`
    - `sg_mismatch`
    - `wyckoff_multiplicity_mismatch`
    - `equiv_site_deviation`
    - `cell_reduction_fail`
  - 每类至少输出最小解释字段（如偏差 RMS 或计数）。
- 3.1.3（可选）Wyckoff-level 约束预留接口
  - 新增占位配置与空实现接口，不阻塞主线；
  - 至少保证 CLI 与 metadata 能记录 `wyckoff_constraint` 开关状态。

### 3.2 review
- 验收阈值：`spacegroup_match_rate >= 0.85`，`spglib_fail_rate <= 0.05`（final 口径）。

---

## phase4（E4：MLIP→DFT 筛选链路）
### 4.1 todo（代码级实施规划，按落地顺序）
- 4.1.1 筛选漏斗脚本化
  - 新增 `twodgen/evaluate/screening_pipeline.py`：
    - 生成 → 几何 gate → CHGNet relax → top-K 多样性采样
  - 强制输出 `screening.csv` 与候选结构目录。
- 4.1.2 能量稳定性统计标准化
  - `eval_samples.py` 增加 energy 分布摘要（median/Q1/Q3）；
  - 按 `energy_available` 分层报告，避免“没跑 relax”污染统计。
- 4.1.3 DFT spot-check 清单导出
  - 新增 `twodgen/evaluate/export_dft_spotcheck.py`：
    - 输入 top-K；
    - 输出 DFT 任务清单与结构文件（K=20–100 可配）。
- 4.1.4 DFT 回填闭环（P3）
  - `export_dft_spotcheck.py` 输出标准目录：
    - `dft_jobs/<job_id>/POSCAR|INCAR|KPOINTS|POTCAR.ref|job.json`
    - `dft_manifest.csv`（`job_id <-> sample_id <-> run_path <-> predicted_energy <-> rank`）
  - 新增 `twodgen/evaluate/import_dft_results.py`：
    - 读取 DFT 输出并回填 `screening.csv` 与聚合摘要。
- 4.1.5 top-K 多样性采样算法冻结（P3）
  - 先按能量取 top-M，再做 fingerprint 上的 farthest-point sampling 选 K；
  - 算法参数写入 `run_metadata.json`，保证复现一致。

### 4.2 review
- 漏斗每一层样本数必须可追踪，且 top-K 选择规则可复现。

---

## phase5（E5：新颖性与多样性）
### 5.1 todo（代码级实施规划，按落地顺序）
- 5.1.1 novelty 与去重评估模块
  - 新增 `twodgen/evaluate/novelty.py`：
    - train 距离分数
    - 聚类去重（可配置阈值）
  - 输出 novelty 分布和去重后计数。
- 5.1.1b novelty 定义冻结（P2）
  - v0 指纹先固定为：`composition + in-plane lattice + RDF/ADF` 轻量组合；
  - 明确 2D canonicalization：
    - 真空轴对齐
    - 原胞规约
    - 原子置换不变处理
  - 在 `metrics_summary.json` 写入 fingerprint 名称与参数。
- 5.1.2 diversity coverage 与 QD 曲线
  - 新增 `twodgen/evaluate/diversity.py`：
    - spacegroup/composition/N_atoms/lattice bins coverage
  - 新增 `twodgen/evaluate/plot_qd.py` 绘制 validity-diversity tradeoff。
- 5.1.3 防塌陷自动检查
  - 新增 `twodgen/evaluate/check_mode_collapse.py`，给出 coverage 退化报警。

### 5.2 review
- 验收口径：有效率提升时，多样性覆盖不出现显著退化。

---

## 论文产物打包（跨 phase）
### 6.1 todo
- 6.1.1 表格与图自动导出
  - 新增 `twodgen/evaluate/export_paper_assets.py`：
    - Table1/2/3
    - Fig2/3/4/5/6 对应数据与图片
- 6.1.2 可复现实验清单
  - 新增 `twodgen/evaluate/repro_manifest.py`，汇总每个实验：
    - commit
    - 配置文件
    - seeds
    - 输入/输出路径
- 6.1.2b 环境锁定信息（P0）
  - `run_metadata.json` 与 `repro_manifest` 增加依赖版本：
    - `python/torch/spglib/ase/pymatgen/chgnet`
  - 记录 device、dtype、可选 CUDA 信息，降低跨机漂移。
- 6.1.3 补充文档
  - 更新 `twodgen/process.md`：新增 E0–E5 当前节点与关键结论；
  - 更新 `twodgen/history.md`：每次完成一个子任务追加条目化记录。

### 6.2 review
- 论文所需图表与表格可由脚本一键重建，且与 run_metadata 可追溯一致。

---

## 本轮补齐完成（2026-02-10）
- phase2（E2）
  - 已在 `twodgen/scrip/train_tokens.py` 增加：
    - `post_project_trigger_rate_train_proxy`
    - `cond_violation_rate_train_proxy`
    - `vacuum_violation_rate_train_proxy`
  - 已新增训练聚合产物：`train_metrics_aggregate.json`（含 proxy 前后半程趋势）。
  - 已新增 `twodgen/scripts/exp_e2_curriculum_repulsion.sh` 与 shim `twodgen/scrip/exp_e2_curriculum_repulsion.sh`。
  - 已新增 `twodgen/evaluate/check_trigger_trend.py` 与 `twodgen/evaluate/collect_e2_curriculum_repulsion.py`。
- phase3（E3）
  - 已新增 `twodgen/configs/bench/E3_1_soft.yaml`、`twodgen/configs/bench/E3_1_hard.yaml`。
  - `twodgen/scrip/train_tokens.py` 与 `twodgen/scrip/sample_tokens.py` 已统一支持 `symmetry_mode`、`wyckoff_constraint` 开关并写入 metadata。
  - `twodgen/evaluate/eval_samples.py` 已补齐：
    - `spacegroup_match_rate`
    - `spglib_fail_rate`
    - `symmetry_violation_breakdown`（固定 5 类枚举）。
- phase4（E4）
  - 已新增 `twodgen/evaluate/screening_pipeline.py`，输出 `screening.csv` 与候选目录。
  - `twodgen/evaluate/eval_samples.py` 能量统计已补齐 `median/q1/q3`，并新增 `energy_available` 分层摘要。
  - 已新增 `twodgen/evaluate/export_dft_spotcheck.py` 与 `twodgen/evaluate/import_dft_results.py`，形成 DFT 导出/回填闭环。
  - top-M + FPS top-K 规则已固定，并写入 `run_metadata.json` / `screening_summary.json`。
- phase5（E5）
  - 已新增：
    - `twodgen/evaluate/novelty.py`
    - `twodgen/evaluate/diversity.py`
    - `twodgen/evaluate/plot_qd.py`
    - `twodgen/evaluate/check_mode_collapse.py`
  - novelty v0 指纹固定为 `composition + in-plane lattice + RDF/ADF`，并支持将 fingerprint 名称/参数写回 `metrics_summary.json`。
- phase6（论文产物）
  - 已新增：
    - `twodgen/evaluate/export_paper_assets.py`
    - `twodgen/evaluate/repro_manifest.py`
  - 已在 `twodgen/common/run_metadata.py` 增加环境锁定信息：
    - `python/torch/spglib/ase/pymatgen/chgnet`
    - `device/dtype/cuda_version/cuda_devices`。
  - `twodgen/configs/bench/experiments.yaml` 已补齐 E2/E4/E5 runner 与 E3 hard_config 引用。

## 需要验证环节（新增）
- E2 训练趋势验证
  - 用真实训练 run 执行 `twodgen/evaluate/check_trigger_trend.py`，确认 3 个 train_proxy 后半程下降。
  - 对比同一 checkpoint 的采样评估，确认 `post_project_trigger_any_rate` 降低且 `success_geom_rate` 不退化。
- E3 对称性验证
  - 分别运行 soft/hard 配置，对比 `spacegroup_match_rate` 与 `spglib_fail_rate`。
  - 抽样检查 `symmetry_violation_breakdown` 5 类字段在 `tier0_metrics.json` 与 `failure_breakdown.json` 一致。
- E4 DFT 链路验证
  - 跑一轮 `screening_pipeline.py -> export_dft_spotcheck.py -> import_dft_results.py`，确认 `dft_manifest.csv` 与 `screening_with_dft.csv` 可回填。
  - 检查 `dft_jobs/*/POSCAR|INCAR|KPOINTS|POTCAR.ref|job.json` 是否齐全。
- E5 多样性/新颖性验证
  - 运行 `novelty.py` 与 `diversity.py` 后，再执行 `check_mode_collapse.py`，确认 coverage/novelty 阈值判定符合预期。
  - 用 `plot_qd.py` 复核 validity-diversity 曲线与输入统计一致。
- 论文产物验证
  - 运行 `export_paper_assets.py` 与 `repro_manifest.py`，确认表/图可一键重建，且 manifest 中 commit/config/seed/环境版本完整可追溯。
