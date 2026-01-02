# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 高（会显著影响训练/评估结论）
（当前无）

## 中（会影响可用性/可复现性）
1. 采样与评估脚本出现双向导入，存在循环依赖风险  
   - 现状：`twodgen/evaluate/eval_samples.py` 导入 `twodgen/scrip/sample_tokens.py`，后者又导入 `twodgen/evaluate/eval_samples.py`。目前运行可通过，但存在初始化顺序不稳的潜在风险。  
   - 相关文件：`twodgen/evaluate/eval_samples.py`、`twodgen/scrip/sample_tokens.py`

2. 评估脚本被动引入重依赖，导致“仅评估 npz”也需要 torch/pymatgen  
   - 现状：`eval_samples.py` 在顶层导入 `sample_tokens`，从而触发 torch/pymatgen 等重依赖加载，即使只是 `--samples` 评估。  
   - 影响：增加评估环境依赖与启动开销。  
   - 相关文件：`twodgen/evaluate/eval_samples.py`

## 低（历史/边缘问题，但建议记录）
1. 预处理脚本对异常样本吞掉异常且不记录数量  
   - 现状：`prepare_c2db_tokens.py` 中 `row_to_tokens(...)` 异常直接 `continue`，没有日志或计数。  
   - 影响：数据损失难以追踪，排查困难。  
   - 相关文件：`twodgen/data/prepare_c2db_tokens.py`

2. 训练 DataLoader 固定 `drop_last=True`，小数据/调试场景可能丢样本  
   - 现状：非 bucket 分支 DataLoader 总是 `drop_last=True`。  
   - 影响：样本量较小或 batch size 大时可能导致有效样本减少。  
   - 相关文件：`twodgen/scrip/train_tokens.py`
