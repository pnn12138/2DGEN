## 冗余清理完成
- 本轮已完成冗余代码清理与合并，无待处理项。
- 发现（2026-01-31）：`twodgen/scrip/sample_tokens.py` 在头部和 `parse_args` 之后分别重定义 `_parse_pbc_mask`，造成维护冗余；建议保留一处并移除另一处。
- 发现（2026-02-05）：`_parse_pbc_mask` 在 `prepare_c2db_tokens.py`、`train_tokens.py`、`sample_tokens.py`、`evaluate/eval_samples.py`、`evaluate/plot_compare.py` 和 `evaluate/eval_tier0_cif.py` 等模块中反复实现，考虑集中到 `twodgen/common` 或 `twodgen/data` 的通用工具防止未来改动不同步。
