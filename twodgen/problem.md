# twodgen 训练-采样问题清单（按严重程度排序）

> 以当前代码/文档为准，按 P0（最严重）……P4（较轻）排列。每个问题都注明影响与定位文件。

## P0：collision 导致 valid rate 被拖垮
- Baseline评估 (`twodgen/baselines/eval_run_001.md` + `eval/tier0_metrics.json`) 复现 valid_rate/valid_2d_rate 极低（~0.2/0.06），原因几乎全部捕捉自 `twodgen/evaluate/eval_samples.py:330-446` 中的 `min_dist < bond_cut` 判定；`fail_reason_counts.collision` 依然占主导。训练端默认只加 0.02 的 collision penalty 并且没有打开 `--filter-min-dist-below`/`--curriculum-collision`（`twodgen/scrip/train_tokens.py:26-120`），因此 collision 样本仍被频繁抽到，采样输出 `min_dist` 分布和有效率没有实质改善。

## P1：coord_frame 回退时仍然混用 canonical geometry
- `C2DBTokenNPZDataset` 会在缺失 `f_canon/gram6_canon` 时设置 `coord_frame_actual="raw"`，但 `uv_angle/z_norm/lattice_param` 等 canonical 字段仍然可用（`twodgen/data/c2db_dataset.py:333-465`）。`twodgen/scrip/train_tokens.py:1003-1020` 在检测到这些字段存在时会默认启用 geometry heads，不再检查 `coord_frame_actual`，导致训练时 raw 的 `frac/gram6` 与 canonical geometry heads/conditions 混用，破坏了条件一致性。

## P2：清洗/标签产物没有接入训练
- `twodgen/data/clean_c2db_2d.py:204-360` 能生成 `c2db_clean_2d.csv`、`c2db_quality.jsonl` 等质量标注，但仓内除了 `twodgen/process.md`/`twodgen/data/README.md` 外再无消费这些文件（`rg -n c2db_quality` 只返回文档），`train_tokens` / `sample_tokens` 仍旧指向 `data/C2DB/cache/c2db_tokens_2d_based.npz`。换言之，质量标签、source bucket 与 hard-pass 筛选结果并未参与 dataset 构造、collison curriculum 或评估，因此清洗工作无法真正保障训练/评估集的质量。

## P3：Tier-2 性质预测只是占位
- `twodgen/evaluate/property_predict.py:1-80` 仅输出常数或随机值来模拟性质预测（`--mock-predict`），没有接入任何真实性质模型或数据。既然 Phase 3 要固化 Tier-2 指标，该脚本仍然是占位，无法提供功能性打分，导致分层评估链路在功能层面空转。

## P4：形成能流水线依赖但缺少参考能量文件
- `twodgen/evaluate/run_pipeline.py:124-140` 与 `twodgen/evaluate/formation_energy.py:30-60` 强制要求 `--ref-energies` JSON，而 README (`twodgen/README.md:102-110`) 举例使用 `data/ref_energies.json`。仓库内无此文件（`find data -name 'ref_energies.json'` 空），也没有生成它的脚本，导致 CIF 评估 pipeline 无法完整跑通形成能阶段。
