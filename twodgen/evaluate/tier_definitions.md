# Tier-0/1/2 评估定义（Phase 3）

本文件用于固化分层评估指标与输出字段规范，便于后续统一报告与图表生成。

## Tier-0：几何有效性（CIF/采样）
- 核心指标：
  - min_dist（最小原子间距）
  - collision_flag（是否碰撞）
  - vacuum / vacuum_ratio / is_2d_flag
- CIF 输入输出字段：
  - id, cif_path, n_atoms, min_dist, collision_flag
  - vacuum, c_len, vacuum_ratio, is_2d_flag, valid, fail_reason

## Tier-1：稳定性与条件匹配
- 条件匹配：
  - formula / elements / spacegroup_number
  - cond_match / fail_reason
- 稳定性（MatterSim + 形成能）：
  - total_energy / energy_per_atom
  - formation_energy_per_atom / formation_pass

## Tier-2：功能性质（占位或真实模型）
- property_key（如 band_gap）
- property_pass（是否达标）

## 合并输出（merge_reports.py）
- per_sample.jsonl（合并后）字段示例：
  - id, cif_path, valid, cond_match, formation_pass, property_pass
  - min_dist, vacuum, vacuum_ratio, is_2d_flag
  - formula, elements, spacegroup_number
  - total_energy, energy_per_atom, formation_energy_per_atom
- report.json 字段：
  - schema_version: merged_report_v1
  - total_samples
  - rates: valid_rate / cond_match_rate / formation_pass_rate / property_pass_rate
