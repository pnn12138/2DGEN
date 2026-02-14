# twodgen 冗余项清单（2026-02-10）

> 目标：只保留主线必需代码。下面条目按“可直接清理价值”排序。

## A. 代码冗余

### 1) `_parse_pbc_mask` 被重复实现（建议统一）
- 重复位置：
  - `twodgen/scrip/train_tokens.py:642`
  - `twodgen/scrip/sample_tokens.py:594`
  - `twodgen/data/prepare_c2db_tokens.py:426`
  - `twodgen/evaluate/eval_samples.py:308`
  - `twodgen/evaluate/eval_tier0_cif.py:13`
  - `twodgen/evaluate/plot_compare.py:12`
- 建议：上移到 `twodgen/common`（例如 `twodgen/common/cli_utils.py`），各入口统一调用。

### 2) 子进程调用样板代码重复
- 位置：`twodgen/evaluate/run_pipeline.py`, `twodgen/evaluate/self_train_loop.py`, `twodgen/evaluate/screening_pipeline.py`
- 问题：重复拼命令、重复 `_run()`、参数转义风格不一致。
- 建议：抽出统一 runner（处理 `sys.executable`、`shlex.split`、日志和错误包装）。

## B. 文件与入口冗余

### 3) `twodgen/scrip/` 与 `twodgen/scripts/` 并存
- 现状：`scripts` 中多个文件只是转发到 `scrip`，同时 `scrip` 里又保留了 deprecated shim。
- 影响：入口路径双轨，文档和自动化脚本容易混用。
- 建议：确定唯一入口目录（建议 `twodgen/scripts/`），保留一个短期兼容窗口后移除另一套。

### 4) 一次性实验脚本仍在主目录
- 典型文件：
  - `twodgen/scrip/debug_cond_trigger.sh`
  - `twodgen/scrip/sampling_projection_ab.sh`
  - `twodgen/scrip/eval_with_energy.sh`
  - `twodgen/evaluate/test_relax.py`
  - `twodgen/evaluate/eval_run_001.py`
- 建议：迁移到 `twodgen/experiments/archive/`（或文档化后删除），避免被误当成生产入口。

## C. 清理优先级建议
1. 先统一 `_parse_pbc_mask`（低风险、收益高）。
2. 再统一入口目录和子进程 runner（降低维护成本）。
3. 最后归档一次性实验脚本（清理噪声，减少误用）。
