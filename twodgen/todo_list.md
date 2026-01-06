# twodgen 问题修复待做项（problem.md 对齐）

## 目标
修复 `problem.md` 中的训练/评估问题，保证几何字段对齐、坐标系一致、评估可靠、脚本可用性稳定。

## 高优先级（训练正确性）
### 1) 采样/评估阈值域对齐（min-dist）
- 目标：让 sampling log 的 `valid_rate` 与评估口径一致，避免误判。
- 方案：
  - 在 `twodgen/scrip/sample_tokens.py` 统一参数命名为 `--eval-min-dist`（默认 1.5A），保留 `--min-dist` 为兼容别名并打印弃用提示。
  - 在 `twodgen/scrip/sample_tokens.py` 保存到 `samples.npz` 的元信息中追加 `min_dist_cut` 与 `valid_rate` 口径说明。
  - 在 `twodgen/evaluate/eval_samples.py` 读取 samples 元信息并打印对齐检查：`eval_min_dist` vs `min_dist_cut`，不一致时给出明确警告。
  - 在 sampling 日志打印：`min_dist_cut`、`eval_min_dist`、`valid_rate` 口径。
- 验收：
  - 同一批 samples 的 sampling log 与 eval log 的 valid_rate 差异仅来自随机性（可用 3 次重复采样验证）。
  - 评估脚本在阈值不一致时明确提示且可追溯到 samples 元信息。

### 2) 采样阶段最小距离约束/后处理
- 目标：降低 collision 为主的失败模式。
- 方案：
  - 在 `twodgen/model/atom_denoiser.py` 的 `_project_step` 或 `generate()` 增加可选后处理钩子。
  - 使用 `twodgen/common/crystal.py` 的 MIC 距离实现，进行轻量 repulsion：
    - 迭代步数 `--min-dist-iter`（默认 0 关闭）
    - 排斥强度 `--min-dist-strength`（默认 0.02~0.05，线性缩放位移）
    - 最小距离阈值 `--min-dist-cut`（默认跟 `--eval-min-dist` 同值）
  - 仅对 `< cut` 的原子对施加位移，位移沿最短位移向量，保留整体质心不漂移。
  - 记录后处理前后 `min_dist` 分布（mean/p10）与 `collision` 计数。
- 验收：
  - 相同 checkpoint/条件下 `collision` 数量显著下降（至少降低 50% 或达到目标阈值）。
  - `min_dist` 分布 p10 接近或高于 `1.5A`。
  - 后处理开启/关闭对 thickness/vacuum 指标影响可忽略。

### 3) project-geometry 的安全护栏
- 目标：避免未训练几何头时的随机扰动。
- 方案：
  - 在 `twodgen/scrip/sample_tokens.py` 读取 checkpoint 的 `geometry_config.use_geometry_fields`：
    - 若 False 且传入 `--project-geometry`，直接报错并提示正确用法。
    - 若 True，打印几何头启用状态与相关 loss 标识（从 checkpoint 元信息读取）。
  - 在日志中明确打印 `project_geometry=True/False` 与几何字段来源。
- 验收：
  - 未训练几何头的 ckpt 无法误启用几何投影。
  - 训练过几何头的 ckpt 采样日志包含清晰的几何投影信息。

### 4) 条件采样索引策略
- 目标：避免默认重复同一条件。
- 方案：
  - 在 `twodgen/scrip/sample_tokens.py` 将默认策略改为 `--cond-random`（当未显式指定 `--cond-index/--cond-first`）。
  - 保留 `--cond-first` 仅用于复现实验，并在日志中打印最终索引策略与采样到的 cond 统计（如 `n_atoms` 分布）。
  - 在 `samples.npz` 的元信息中记录 `cond_strategy` 与采样到的 `cond_indices`。
- 验收：
  - 默认采样时 `n_atoms` 分布不恒定，cond 统计多样化。
  - 评估时能从元信息定位实际 cond 策略。

### 5) order_idx 近简并不连续的稳健性
- 目标：降低同元素原子靠近时的序列不稳定影响。
- 方案：
  - 在 `twodgen/data/preprocess.py` 的排序逻辑中加入稳定 tie-break：
    - 优先使用 `original_idx` 作为最后排序键，避免近简并翻转。
    - 可选：为排序引入极小扰动（固定 seed），仅用于 tie-break，不改变物理坐标。
  - 在评估统计中新增同元素最近邻的 `min_dist` 统计（p10/p50/p90）。
  - 观察是否需要引入 permutation-invariant loss（成本较高，后置）。
- 验收：
  - 同元素最近邻 `min_dist` 低端尾部改善，并在评估报告可见。
  - 排序稳定性提升后训练/采样表现更一致。

## 中优先级（可用性/评估稳定）
### 6) 评估统计与日志补强
- 目标：快速定位失败来源，减少“只看 valid_rate”的信息缺口。
- 方案：
  - 在 `twodgen/evaluate/eval_samples.py` 输出 `collision` 的最小距离分布（min/mean/p10）。
  - 输出 `n_atoms` 与元素分布的基本统计，便于发现条件采样异常。
  - 汇总关键阈值与参数（`eval_min_dist`、`thickness/vacuum` 阈值）。
- 验收：评估日志包含完整诊断信息，可直接定位失败类型。

## 低优先级（体验/健壮性）
### 7) CLI 参数一致性与帮助文案
- 目标：降低脚本误用成本。
- 方案：
  - 统一 `sample_tokens.py` 与 `eval_samples.py` 的参数命名与默认值。
  - 帮助文案中明确 “采样过滤阈值 != 评估阈值” 的历史差异，避免误读。
- 验收：帮助文案可以直接指导正确用法。

## 文档补齐
### 8) README/内部说明补齐
- 目标：保证复现流程清晰。
- 方案：
  - 在 `twodgen/problem.md` 追加最新修复点与推荐采样命令。
  - 记录 `project-geometry` 的启用条件与示例。
- 验收：新同学可按文档复现评估流程。

## 依赖与顺序建议
推荐顺序：1 -> 3 -> 4 -> 6 -> 2 -> 5 -> 7 -> 8
（先对齐评估口径与护栏，再补强评估诊断，随后做防碰撞与排序稳定，最后统一文档与 CLI）。
