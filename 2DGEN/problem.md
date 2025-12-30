# 2DGEN 训练-采样问题清单（按严重程度排序）

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
