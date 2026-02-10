## 冗余清理完成
- 本轮已完成冗余代码清理与合并，无待处理项。
- 发现（2026-01-31）：`twodgen/scrip/sample_tokens.py` 在头部和 `parse_args` 之后分别重定义 `_parse_pbc_mask`，造成维护冗余；建议保留一处并移除另一处。
- 发现（2026-02-05）：`_parse_pbc_mask` 在 `prepare_c2db_tokens.py`、`train_tokens.py`、`sample_tokens.py`、`evaluate/eval_samples.py`、`evaluate/plot_compare.py` 和 `evaluate/eval_tier0_cif.py` 等模块中反复实现，考虑集中到 `twodgen/common` 或 `twodgen/data` 的通用工具防止未来改动不同步。

## 冗余调试文件待清理（2026-02-09）
- `twodgen/scrip/debug_cond_trigger.sh`：Phase1b 专用短跑对照脚本，验收完成后可移至归档或删除。
- `twodgen/scrip/sampling_projection_ab.sh`：采样投影 A/B 对照脚本，当前仅用于一次性验证。
- `twodgen/scrip/eval_with_energy.sh`：采样+relax+eval 的一键脚本，属于临时实验入口。
- `twodgen/evaluate/test_relax.py`：CHGNet relax 的手工 smoke test，非正式测试用例。
- `twodgen/scrip/test_tokens.py`：旧的 token diffusion smoke test，功能与 pytest 覆盖重复。
- `twodgen/evaluate/eval_run_001.py`：基线复现实验 runner（可保留，但若不再维护建议归档到 `baselines/` 或文档化后移除）。
