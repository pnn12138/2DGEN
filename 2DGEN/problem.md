# 2DGEN 训练-采样问题清单（按严重程度排序）

## 高（会显著影响训练/评估结论）
（暂无）

## 中（会影响可用性/可复现性）
1. `main.py` 提示的训练/采样命令路径不正确  
   - 实际脚本位于 `2DGEN/scrip/`，当前提示容易误导新手无法直接运行。
2. 评估指标未与训练/采样日志打通  
   - `evaluate/eval_samples.py` 已实现 Tier‑0/1 指标，但未集成到训练/采样流程，缺少系统化记录。
3. 关键断言与邻居构建单测缺失  
   - `guide.md` 建议的 finiteness/索引边界断言与邻居‑mask 单测未实现，仅有 smoke test。

## 低（历史/边缘问题，但建议记录）
1. `cholesky6` 的 clamp 尺度单位错误会导致晶胞长度塌缩到 ~41Å（已修复）  
   - 现已在 `scrip/train_tokens.py` 与 `scrip/sample_tokens.py` 修正为“内部尺度（物理长度 / sqrt(g_scale)）”统计与 clamp。  
   - 若复现到旧 checkpoint/旧脚本，仍可能遇到同类问题。
