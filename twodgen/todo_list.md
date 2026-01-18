# twodgen Phase 3 评估方案（未落地部分）

> 目标：完成 plan.md 的分层评估链路（CIF → 条件 → MatterSim → 形成能 → 性质预测 → 报告合并），并补齐消融与论文级图表。

## P0：评估链路主闭环（优先）
- 增加 CIF 入口评估脚本 `twodgen/evaluate/eval_tier0_cif.py`：
  - 解析 CIF（ASE/Pymatgen），输出 min_dist/碰撞/真空/2D 判定字段。
  - 支持 `vacuum_ratio` 判据与 `is_2d_flag`。
- 增加条件校验脚本 `twodgen/evaluate/check_conditions.py`：
  - 配方/元素集合/空间群匹配检查。
  - 输出 `cond_match` 和失败原因统计。
- 增加 MatterSim 能量评估 `twodgen/evaluate/mattersim_energy.py`：
  - CIF → ASE → 能量（可选 relax）。
  - 输出 `total_energy` / `energy_per_atom` / `composition`。
- 增加形成能计算 `twodgen/evaluate/formation_energy.py`：
  - 读取元素参考能表，输出 `formation_energy_per_atom` 与通过率。
- 增加报告合并 `twodgen/evaluate/merge_reports.py`：
  - 合并 Tier-0/条件/能量/形成能输出到统一 `per_sample.jsonl` + `report.json`。
- 增加一键 pipeline `twodgen/evaluate/run_pipeline.py`：
  - 串联上述步骤，支持 `--pipeline-steps` 部分执行。

## P1：Tier-2 性质预测（可占位）
- 增加 `twodgen/evaluate/property_predict.py`：
  - 先提供 `--mock-predict`（常数/随机值）打通流程。
  - 后续替换为真实性质模型（带隙/模量/磁矩）。

## P2：输出规范与文档
- 更新 `twodgen/evaluate/tier_definitions.md`：
  - 明确 CIF 入口字段、合并报告 schema、字段版本号。

## P3：统计与图表（论文级）
- 形成能分布图脚本 `twodgen/evaluate/plot_forme_distribution.py`。
- 评估结果对比脚本 `twodgen/evaluate/compare_scenarios.py`。
- SUN 指标统计 `twodgen/evaluate/sun_metrics.py`（先用简单 unique/novel）。

## P4：消融实验支持
- 消融矩阵配置（JSON）与运行脚本（如 `run_ablation.py`）。
- 生成对比表：valid/cond_match/formation_pass 率。
