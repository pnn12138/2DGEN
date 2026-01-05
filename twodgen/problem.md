# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 高（会显著影响训练/评估结论）
1. `uv_angle/z_norm` 与 `z/frac` 的原子顺序不一致，几何扩散信号被错配 ✅  
   - 修复：缓存双写并引入 `order_idx/order_inv`，训练端支持 `--align-atoms` 自动重排 per‑atom 字段；`uv_angle/z_norm` 与 `z/frac/atom_mask` 对齐。  
   - 备注：需重新生成 npz 才会生效（推荐 `--coord-frame canon`）。
   - 相关文件：`twodgen/data/preprocess.py`、`twodgen/data/prepare_c2db_tokens.py`、`twodgen/data/c2db_dataset.py`、`twodgen/scrip/train_tokens.py`

2. `order_idx` 虽保存但从未用于对齐，几何字段无法纠正顺序 ✅  
   - 修复：`C2DBTokenNPZDataset` 支持基于 `order_idx` 对齐输出，并新增护栏检查与日志提示。  
   - 备注：老缓存需迁移/重生成。  
   - 相关文件：`twodgen/data/c2db_dataset.py`、`twodgen/scrip/train_tokens.py`

3. 坐标系不一致：`uv_angle` 基于 canonical 2D 基底，而 `frac_coords` 仍在原晶格基底 ✅  
   - 修复：方案 A 落地，新增 `f_canon/lattice_canon/gram6_canon`；`frac_canon` 由 `x = f_raw @ L_raw` 与 `f_canon = x @ L_canon^{-1}` 构造，并同步 `u_shift/v_shift`。  
   - 备注：训练/采样可用 `--coord-frame canon`；评估对比可在 `plot_compare` 中选择 coord frame。  
   - 相关文件：`twodgen/data/preprocess.py`、`twodgen/data/prepare_c2db_tokens.py`、`twodgen/data/c2db_dataset.py`、`twodgen/scrip/train_tokens.py`、`twodgen/scrip/sample_tokens.py`

## 中（会影响可用性/可复现性）
1. 采样与评估脚本出现双向导入，存在循环依赖风险 ✅  
   - 修复：`eval_samples.py` 改为惰性导入采样模块，仅 `--sample` 分支加载。  
   - 相关文件：`twodgen/evaluate/eval_samples.py`

2. 评估脚本被动引入重依赖，导致“仅评估 npz”也需要 torch/pymatgen ✅  
   - 修复：`--samples` 评估路径不再加载采样模块与重依赖。  
   - 相关文件：`twodgen/evaluate/eval_samples.py`

3. 对比绘图使用近似 MIC（rounding），在非正交晶格下距离评估不可靠 ✅  
   - 修复：`plot_compare` 默认使用精确 MIC，保留 `--mic-mode approx` 兼容旧图；支持 `--coord-frame canon`。  
   - 相关文件：`twodgen/evaluate/plot_compare.py`

## 低（历史/边缘问题，但建议记录）
1. 预处理脚本对异常样本吞掉异常且不记录数量 ✅  
   - 修复：统计跳过原因并写入 `preprocess_stats.json`，支持 `--verbose` 打印示例错误。  
   - 相关文件：`twodgen/data/prepare_c2db_tokens.py`

2. 训练 DataLoader 固定 `drop_last=True`，小数据/调试场景可能丢样本 ✅  
   - 修复：新增 `--drop-last/--no-drop-last`。  
   - 相关文件：`twodgen/scrip/train_tokens.py`

3. 默认输出路径疑似拼写错误（`ache` vs `cache`） ✅  
   - 修复：默认输出改为 `data/C2DB/cache/c2db_tokens.npz`，文档同步。  
   - 相关文件：`twodgen/data/prepare_c2db_tokens.py`、`twodgen/README.md`
