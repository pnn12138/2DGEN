# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 高（会显著影响训练/评估结论）
（当前无）

## 中（会影响可用性/可复现性）
1. `prepare_dataloader(..., shuffle=...)` 的 `shuffle` 参数在非 bucket 分支被忽略  
   - 现状：`prepare_dataloader()` 在 `use_buckets=False` 时固定 `shuffle=True`，不使用传入参数。  
   - 影响：命令行/调用侧即使想关闭 shuffle 也无效（调试/复现时不方便）。  
   - 相关文件：`twodgen/scrip/train_tokens.py`

2. checkpoint 加载使用 `torch.load(..., weights_only=False)`（安全边界较弱）  
   - 现状：`sample_tokens.py` 直接反序列化 checkpoint 对象。  
   - 影响：如果 checkpoint 来源不可信，`torch.load` 可能触发任意代码执行风险；建议至少在文档中注明“仅加载可信 checkpoint”，或评估迁移到 `weights_only=True` + 显式 config。  
   - 相关文件：`twodgen/scrip/sample_tokens.py`

3. 评估指标未与训练/采样日志打通  
   - `evaluate/eval_samples.py` 已实现 Tier‑0/1，但训练/采样默认流程里没有自动调用与归档（仍是手工步骤）。

4. 关键断言与邻居构建单测缺失  
   - `guide.md` 中提到的 finiteness/索引边界断言与邻居‑mask 单测未落地，目前主要依赖 `scrip/test_tokens.py` 的 smoke test。

5. 可执行入口依赖“从仓库根目录运行”，缺少可安装/可复用的 CLI  
   - 现状：当前主要通过 `uv run python -m twodgen.scrip.*` 在 repo root 下运行；未提供 `console_scripts`/`python -m twodgen` 的统一入口。  
   - 影响：在别的工作目录、或作为依赖复用时不够稳；也不利于 IDE/单测发现入口。  
   - 相关文件：`pyproject.toml`、`twodgen/scrip/*.py`

## 低（历史/边缘问题，但建议记录）
1. `__pycache__/` 与 `.pyc` 出现在仓库树中（应由 `.gitignore` 排除）  
   - 影响：污染 diff、误导代码审查；在不同 Python 版本间还会产生噪声冲突。  
   - 相关路径：`twodgen/__pycache__/`、`twodgen/common/__pycache__/`、`twodgen/data/__pycache__/`、`twodgen/model/__pycache__/`、`twodgen/scrip/__pycache__/`

2. `C2DBMetadata.space_group_number` 类型注解与实际写入不一致  
   - 现状：字段标注为 `Optional[int]`，但 `_coerce_optional_float()` 返回 `Optional[float]`。  
   - 影响：主要是类型/可读性问题，运行一般不受影响。  
   - 相关文件：`twodgen/data/c2db_dataset.py`

3. `C2DBDataset.collate_fn` 的类型注解与实现不一致  
   - 现状：注解为 `Iterable[...]` 但实现使用 `batch[0]`；严格来说应是 `Sequence[...]`。  
   - 相关文件：`twodgen/data/c2db_dataset.py`

4. 代码/配置中存在未接入主流程的参数与逻辑（可能误导）  
   - 例：`common/crystal.py:clip_lattice()` 以及 `AtomDenoiserConfig` 的 `v_min/v_max/cond_max` 当前训练/采样主流程未实际使用。  
   - 影响：读代码时容易以为有“晶胞投影/clip”保护，但实际不生效；建议要么接入（如 `--project-each-step` 路径内），要么移除/标注。  
   - 相关文件：`twodgen/common/crystal.py`、`twodgen/model/atom_denoiser.py`、`twodgen/tm.md`

5. `guide.md`（A+++ v3.2）包含多处“计划/建议项”未在代码中落地  
   - 例：双图邻居（`kNN(d_xy)` + `kNN(d_3d)`）、wrap shift 的 edge embedding、排序稳定性单测等。  
   - 影响：新读者容易误判“当前实现已具备这些能力”；建议在文档中明确“已实现/规划中”的边界。  
   - 相关文件：`twodgen/guide.md`
