# twodgen 问题清单（当前未修复，2026-02-10）

> 说明：本文件只记录“当前仍未修复”的代码问题。

## 当前状态
- 当前未修复问题：**0**

## 本轮已修复（2026-02-10）
1. 修复 `twodgen.scrip.train_tokens --help` 崩溃（`argparse` `%` 转义）。
2. 修复 `twodgen.evaluate.eval_tier0_cif` 导入旧私有符号导致入口不可用。
3. 删除 `eval_samples.py` 不可达内置采样分支，并恢复 `--min-dist/--bond-cut/--dup-eps/--v-min/--v-max/--pbc-mask` 显式 CLI 参数。
4. 修复 `twodgen/scrip/eval_with_energy.sh` 的 `--post-project-interval` 语义冲突（`0 -> 1`）。
5. 修复 `train_tokens.py` 和 `test_tokens.py` 的脚本直运行路径注入顺序。
6. 修复 `self_train_loop.py` 参数拆分（`.split()` -> `shlex.split()`），并统一子进程解释器为 `sys.executable`。
7. 修复 `run_pipeline.py` 的 merge 步骤对 energy 产物的硬依赖（改为按文件存在与步骤组合判定），并统一子进程解释器为 `sys.executable`。

## 验证结果
- `uv run pytest -q`：`37 passed`。
- `uv run python -m compileall -q twodgen/evaluate/eval_samples.py twodgen/evaluate/eval_tier0_cif.py twodgen/evaluate/self_train_loop.py twodgen/evaluate/run_pipeline.py twodgen/scrip/train_tokens.py twodgen/scrip/test_tokens.py`：通过。
- 入口验证：
  - `uv run python -m twodgen.scrip.train_tokens --help`：通过。
  - `uv run python -m twodgen.evaluate.eval_tier0_cif --help`：通过。
  - `uv run python twodgen/scrip/train_tokens.py --help`：通过。
  - `uv run python twodgen/scrip/test_tokens.py`：通过。
  - `python -m ... --help` 全量体检（`twodgen/scrip/*.py` + `twodgen/evaluate/*.py`）：`40/40` 通过。
