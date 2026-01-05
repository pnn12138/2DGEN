# twodgen 问题修复待做项（problem.md 对齐）

## 目标
修复 `problem.md` 中的训练/评估问题，保证几何字段对齐、坐标系一致、评估可靠、脚本可用性稳定。

## 高优先级（训练正确性）
### 1) 几何字段顺序对齐（uv_angle/z_norm 与 z/frac）✅
- 目标：保证 uv_angle/z_norm 与 z/frac/atom_mask 同一原子顺序。
- 方案 A（推荐）：在缓存或加载时使用 `order_idx` 统一重排 `z/frac/atom_mask` 到 canonical 顺序。
  - `prepare_c2db_tokens.py`（已实现）：
    - 追加 `schema_version`（如 `v4`) 与 `coord_frame` 字段写入 npz。
    - 双写字段：`z_raw/f_raw/atom_mask_raw` + `z_canon/frac_canon/atom_mask_canon`（或 `z/f/atom_mask` + `*_canon`），默认保留旧字段避免破坏旧脚本。
    - 新增 `order_inv`（`argsort(order_idx)`）并写入 npz，便于原/规范双向映射。
    - 统一对齐：所有 per-atom 字段在 canonical 分支一起重排（`u/v/z_norm/uv_angle/uvz` 等）。
  - `c2db_dataset.py`（已实现）：
    - 读取 `schema_version/coord_frame`，决定输出 raw or canon。
    - 若 `order_idx` 存在但只提供 raw 字段，按 `order_idx` 动态重排；同时保留 `order_idx/order_inv`。
    - 增加 `align_atoms` 配置（dataset 级/训练 CLI）控制输出顺序。
  - `train_tokens.py`（已实现）：
    - 数据加载后一次性提示：是否启用对齐、使用的 `coord_frame` 与 `schema_version`。
    - 如果 `use_geometry_fields` 且 `order_idx` 存在但未对齐，直接报错或强提醒。
- 方案 B：在预处理阶段把 uv_angle/z_norm 反排回原始 CIF 顺序（生成 `uv_angle_raw/z_norm_raw`），保持 `z/f` 不变。
- 输出：统一对齐的字段；老缓存需要迁移或重新生成（补充迁移脚本/说明）。
  - 注意：凡是 per-atom 字段（1D/2D/3D）都要一起重排，避免隐蔽错配（如 one-hot、标签、邻居索引/边特征等）。
  - 建议：双写一段时间（保留 *_raw 与 *_canon 或显式 `schema_version/coord_frame`），训练按配置选择。

### 2) order_idx 真正参与对齐 ✅
- 目标：不再“只保存不使用”。
- 实现：
  - 在 `C2DBTokenNPZDataset.__getitem__` 内添加可选对齐分支（基于配置或缓存版本字段）。
  - 增加 `order_inv = argsort(order_idx)` 以便双向映射。
  - 训练日志加入一次性提示：当前批次是否启用对齐（避免默默错误）。
  - 护栏断言：若存在 `order_idx` 但最终输入仍为 raw 顺序，直接报错或强提醒。
  - Debug 断言：抽样重算/对比 `uv_angle/z_norm` 统计，验证重排生效。
  - 覆盖范围：对齐必须覆盖 `z/f/atom_mask`，以及 `u/v/uvz/uv_angle/z_norm` 与任何新增 per-atom 字段。

### 3) 坐标系一致性（uv_angle/z_norm vs frac）✅
- 问题：uv_angle 由 canonical 2D 基底构造，而 frac 仍在原晶格基底。
- 方案 A（推荐）：引入 `frac_canon`（由 `a_hat/b_hat/n` 构造的 canonical lattice）并让模型使用 `frac_canon`。
- 方案 B：重写几何特征，使 uv_angle/z_norm 基于原晶格 frac（直接用 frac 的 x/y；z 由 c 方向投影）。
- 需要确认：采样与评估是否要求使用 canonical frac（可能需要更新投影/反变换）。
- 输出：新增字段/配置开关；说明迁移策略。
  - 记录元数据：`coord_frame=raw|canon`、`schema_version`、必要的变换信息。
  - 闭环：采样/评估/写 CIF 全链路同步更新，避免“训练在 canon、评估在 raw”。
  - 具体实现（方案 A）：
    - 预处理保存 canonical lattice（`lattice_canon/gram6_canon`）与 `frac_canon`，并应用 `u_shift/v_shift`。
    - `sample_tokens.py` 输出 canonical lattice + frac，并在 samples.npz 标注 `coord_frame=canon`。
    - 评估使用 samples.npz 内坐标系；plot_compare 支持 `--coord-frame canon` 读取 dataset 的 `f_canon/lattice_canon`。

## 中优先级（可用性/评估稳定）
### 4) 解除 eval/sample 循环依赖 + 轻量评估 ✅
- 将 `eval_samples.py` 对 `sample_tokens.py` 的导入改为惰性导入（仅 `--sample` 分支）。
- 或抽出共享逻辑到 `twodgen/evaluate/sample_runner.py`，两边只依赖轻量模块。
- 目标：`--samples` 模式不拉起 torch/pymatgen 依赖。
  - 具体实现：
    - `eval_samples.py` 顶层删除 `sample_tokens` 导入；在 `if args.sample:` 分支内导入。
    - 若需要共享解析/运行逻辑，抽到 `twodgen/evaluate/sample_runner.py`。

### 5) plot_compare MIC 精确化 ✅
- 替换 `df - round(df)` 近似 MIC。
- 方案：复用 `twodgen.common.crystal.frac_mic_dist` 或 `eval_samples._min_dist_and_shifts` 的精确枚举实现。
- 增加 `--mic-mode {exact,approx}` 以便兼容旧结果。
  - 2D slab 提醒：确保 MIC 使用 PBC mask（`1,1,0`），z 方向不 wrap。
  - 具体实现：
    - 新增 `_min_dist_exact(frac, lattice, pbc_mask)`（复用 common 或 eval 逻辑）。
    - CLI 增加 `--mic-mode` 与 `--pbc-mask` 透传。
    - 默认 `exact`，保留 `approx` 仅用于对比旧图。

## 低优先级（体验/健壮性）
### 6) 预处理异常统计与日志 ✅
- 在 `prepare_c2db_tokens.py` 中统计异常数与跳过原因（空 CIF、解析失败、超原子数等）。
- 输出到 stdout 或写入 `out_dir/preprocess_stats.json`。
  - 具体实现：
    - 计数器：`skipped_empty/parse_error/too_many_atoms/other_error`。
    - 记录总行数、成功数、各类跳过比例。
    - 若 `--verbose` 列出前 N 条错误示例。

### 7) DataLoader drop_last 可配置 ✅
- 为 `train_tokens.py` 增加 `--drop-last/--no-drop-last`，默认保持当前行为但提示小数据风险。
  - 具体实现：
    - CLI 新增参数，传入 `prepare_dataloader`。
    - 当 `--no-drop-last` 且数据量小于 batch size 时打印 warning。

### 8) 默认输出路径修正 ✅
- 将 `data/C2DB/ache/c2db_tokens.npz` 改为 `data/C2DB/cache/c2db_tokens.npz`。
- 同步 README/guide 示例，避免路径误导。
  - 具体实现：
    - 修改 `prepare_c2db_tokens.py` 默认参数。
    - 更新 `twodgen/README.md`、`twodgen/guide.md` 示例路径。

## 文档补齐 ✅
- README/guide 同步新开关：`--use-geometry-fields` / `--project-geometry` / `--dual-graph` / `--wrap-embed-dim`。
  - 具体实现：
    - 训练/采样示例补齐对应 CLI。
    - 简述用途与默认值，避免误用。

## 依赖与顺序建议
已全部落地，当前无需继续拆分步骤。
