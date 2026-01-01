# 2DGEN 训练-采样问题清单（按严重程度排序）

## 高
1. 评估逻辑仍是 3D PBC，未对齐 slab 的 2D PBC  
   - `evaluate/eval_samples.py` 使用 3D wrap 计算 MIC 与最小距离，和训练/采样的 `pbc_mask=1,1,0` 不一致。  
   - 这会导致评估指标偏差，尤其是最小距离/跨真空统计。
2. 预计算邻居索引来自 A++ v3 的 (u,v,z_norm)，但训练中距离用 `frac+lattice` 重算  
   - `prepare_c2db_tokens.py` 的 `nbr_idx` 基于 slab canonical 坐标。  
   - `AtomTransformer` 重算距离时仍用 `frac_mic_dist(frac, lattice)`，两者的几何基底可能不一致。  
   - 结果是 edge_index 与 edge_attr 的几何含义不匹配，影响训练稳定性。

## 低
1. Token 采样的原子数在一个 batch 内固定，降低多样性  
   - `sample_tokens.py` 中若未指定 `--num-atoms`，会从统计分布采一次并用于整个 batch，导致一批样本原子数完全一致。  
   - 不是正确性 bug，但会限制生成多样性。
2. `main.py` 提示的训练/采样命令路径不正确  
   - 实际脚本位于 `2DGEN/scrip/`，当前提示容易误导新手无法直接运行。
3. 评估指标未与训练/采样日志打通  
   - `evaluate/eval_samples.py` 已实现 Tier‑0/1 指标，但未集成到训练/采样流程，缺少系统化记录。
4. 关键断言与邻居构建单测缺失  
   - `guide.md` 建议的 finiteness/索引边界断言与邻居‑mask 单测未实现，仅有 smoke test。
