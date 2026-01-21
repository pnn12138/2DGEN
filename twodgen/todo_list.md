# 待做任务清单

> 说明：这里只放“可直接落代码”的事项（不包含“跑完整训练”这类执行任务）。每条包含：优先级、状态、标题、实现描述。

## 训练稳定性与损失/约束

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码） |
|---|---|---|---|
| P0 | DONE | 训练侧 `loss_cond` 真实生效 | 梳理 `twodgen/scrip/train_tokens.py` / `twodgen/model/atom_denoiser.py` 中 `loss_cond` 产生路径：若当前未计算（日志长期为 0），则实现基于 lattice Gram condition number 的惩罚（例如 `cond = λ_max/λ_min`，对 `log(cond)` 超过阈值部分做 hinge），并把 `loss_cond`/`pred_cond_mean` 明确写入 `metrics`。 |
| P0 | DONE | 用软约束替代 cholesky “硬贴边” | 在 `twodgen/model/atom_denoiser.py::_project_step` 中把 `cell[:, :3]` 的 hard clamp 方案改为“软惩罚 + 轻投影”（例如保持 clamp 但增加 `loss_chol_bound`/或在采样时引入 boundary repulsion），并新增日志字段（训练/采样）统计离边界距离分布，降低 `chol_log_clamp_rate` 长期接近 1 的情况。 |
| P0 | DONE | 引入 lattice 方向的碰撞纠正 | 当前采样 repulsion 主要挪 `frac`，在 lattice 过小/贴边时无能为力；实现 `expand-on-collision` 的“训练侧版本”：当 batch 中 `min_dist < cut` 时，对 lattice 做各向同性放大（或仅在 vacuum 轴放大）得到可微 penalty（例如对 `min_dist` 与 lattice scale 的耦合做 proxy），至少让训练梯度能把 lattice 拉大。 |
| P0 | TODO | 真空约束改为“轴选择一致 + 可微 penalty” | 训练时明确采用与评估一致的 vacuum axis 选择逻辑（最长轴为 vacuum axis），并实现可微 vacuum penalty：基于 `c_len - thickness` 的 softplus/hinge（参考 `prepare_c2db_tokens._thickness_vacuum` 的定义，做 batch 版 torch 实现），记录 `loss_vacuum` 的分解项（`vacuum`, `thickness`, `c_len`）。 |
| P0 | DONE | 梯度/权重调度：loss 动态平衡 | 新增 `twodgen/loss/schedule.py`：为 `loss_weight_dict`/`lambda_*` 提供可组合的动态调度（如 warm-up：`lambda_vacuum(t)=min(1,t/T)`；或不确定性权重 multi-task loss）；在 `train_tokens.py` 增加开关/参数（T、启用项、初始权重），并在日志记录每个 step 的“有效权重”与（可选）各项梯度范数，避免训练早期 vacuum/collision 过度施压。 |
| P1 | TODO | 训练侧 cross-vacuum 风险惩罚 | 增加一个近似可微的 cross-vacuum proxy：对 vacuum axis（`pbc_mask[c_idx]=0`）计算 3D MIC 下跨 cell shift 的近邻（可用 `frac_mic_dist_with_shifts`），若存在 `|shift_c|>0` 且距离 < bond_cut，则增加 penalty；同时在日志中记录 `cross_vacuum_proxy_rate` 以对齐 `eval_samples` 的 `cross_vacuum_rate`。 |
| P1 | TODO | 训练动态诊断仪表板（TB/W&B） | 在训练端引入 `tensorboardX` 或 `wandb`：实时记录 `min_dist`/`vacuum_gap` 分布、`chol_diag`(loghist)、各 loss 曲线（vacuum/lattice/min_dist/cond）及其梯度大小（或梯度范数 proxy）；目标是在前 1 万 step 快速发现 lattice collapse/无真空等问题，并与 JSONL 指标保持一致字段命名。 |

## 条件控制与对称性

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码） |
|---|---|---|---|
| P0 | DONE | 让 `cond_match` 评估有意义 | 排查 `sample_tokens.py` 写入的 `cond_counts_vector` 是否可能被“从生成结果回填”导致 `exact_match_rate=1.0` 失真；若存在，修正为“记录采样时使用的目标条件”，并在 `eval_samples.py` 增加 sanity check（如目标 counts 与生成 counts 的差异分布非恒 0）。 |
| P1 | TODO | 空间群条件：数据侧写入 + 模型侧注入 | 在 `prepare_c2db_tokens.py` 里从 CIF/CSV 元信息写入 `space_group_number`（或离散 `spacegroup_id`）到 npz；在训练侧扩展 `cond_fields`（one-hot/embedding）并在 `AtomTransformer` 增加对应的条件投影（或 Adapter），支持采样时指定 `--target-spacegroup`。 |
| P1 | TODO | 对称性条件硬控制（映射标签 + residual loss） | 在 `prepare_c2db_tokens.py` 新增“对称映射标签”字段（如 mirror/rotation pairs、对称复映索引或 Wyckoff 相关标签，来源可用 spglib）；在 `twodgen/model/atom_denoiser.py` 增加 `symmetry_residual_loss`（对称映射后的坐标残差/等价约束），并让采样/评估端可输出 `symmetry_violation` 统计，作为空间群 soft 条件的“硬补丁”。 |
| P1 | TODO | EGNN/GVP/TFN 等变图层模块（可选） | 新增 `twodgen/model/egnn.py`（或等变 tail 模块）：以 frac 相对坐标建图（kNN/半径），输出更新的 per-atom features/coords（或对 `uvz` 做等变 refinement）；与 `AtomDenoiser.forward()` 解耦，通过配置选择 Transformer-only vs Transformer+equivariant tail，并提供最小 smoke test 保障前向/反向可跑。 |
| P2 | TODO | EGNN Tail：Adapter/轻量过渡模块 | 为 “Transformer + EGNN Tail” 预留 adapter（如线性投影/门控残差）以把 Transformer hidden states 映射到 EGNN node/edge features；配置支持逐步开启 tail（冻结/微调主干），并补齐 ablation 配置（no-tail / adapter-only / full-tail）。 |
| P2 | TODO | EGNN Tail 对比：GVP/TFN 备选实现 | 在 `twodgen/model/` 增加 GVP 或 TFN 的最小 tail（接口对齐 EGNN tail），并提供统一的 tail registry/config，便于对比实验结构与指标（有效率、对称性违约、lattice collapse）。 |
| P1 | TODO | 空间群匹配评估（spglib） | 在 `evaluate/` 新增/补齐 `spacegroup_match`：对生成 CIF/structure 调用 spglib 得到预测 spacegroup，与目标 `spacegroup_id` 对比，输出 `spacegroup_match_rate`、按 group 的 breakdown，并写入 `tier1` 或独立 `tier_spacegroup_metrics.json`。 |
| P1 | TODO | 条件控制接口统一 | 在 `sample_tokens.py` 增加统一的条件入口（例如 `--cond-npz` + `--cond-filter` 支持 formula/spacegroup/vacuum bucket），并把筛选规则、抽样策略、最终使用的条件写入 `samples.npz` 与 `per_sample.jsonl`，保证可复现。 |
| P2 | TODO | 组合条件重参数化：联合控制接口 | 设计组合条件抽样器/接口（如 vacuum bucket + composition + spacegroup 的联合筛选与优先级规则），并把该逻辑固化在 `sample_tokens.py` 的单一入口，输出可复现的 `cond_manifest.json`/写回 `samples.npz` 元数据。 |
| P2 | TODO | FiLM-style 条件注入路径扩展 | 在 `AtomTransformer` 增加可选 FiLM/AdaLN 条件注入（按层/按 head），支持组合条件（向量拼接只是 baseline）；提供开关以与现有 one-hot/embedding 注入并行对比。 |

## 评估缓存与稳定性闭环

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码） |
|---|---|---|---|
| P1 | TODO | 统一 “评估输出命名” 的读取入口 | 新增一个小工具模块（例如 `twodgen/evaluate/io.py`）统一解析 eval 目录：优先读取 `tier0_metrics.json`/`tier1_2d_metrics.json`/`per_sample.jsonl`，若发现旧命名则提示并自动迁移；让所有绘图/统计脚本只依赖该入口，避免再次出现命名漂移。 |
| P1 | TODO | 训练/采样/评估：加入 “lattice collapse” 诊断包 | 在 `eval_samples.py` 里补充更直接的 lattice 诊断：输出 `chol_diag`（或其 log）分位数、`volume` 与 `chol_log_min/max` 距离的分位数，并在采样端的 `samples.npz` 中保存必要的中间量（例如 `cell_cholesky6`），让问题定位不依赖反推。 |
| P1 | TODO | 评估缓存机制 + 结构标签传播 | 新增 `twodgen/evaluate/cache.py`：为 `samples.npz` 的逐结构评估提供缓存与并发（key 可用 structure hash）；在 `samples.npz`/`per_sample.jsonl` 添加/回填 `eval_tag(valid/invalid)`、`collapse_flag`、`spacegroup_mismatch` 等字段，支持 GUI/筛查脚本快速定位“高能+碰撞+错组分/错对称”样本。 |
| P1 | TODO | 增加统一指标：`success_rate` | 在 `evaluate/eval_samples.py`（或统一 metrics 汇总处）将 “无碰撞 + vacuum 合法 + 组成准确（exact match）+ energy 稳定（若可用）” 合成为 `success_rate`；同时输出各子条件的 rate 与 per-sample boolean，确保后续 plot/auto_eval 可直接按该字段排序与筛选。 |
| P1 | TODO | 自动形成能估计 + 重排序输出 top‑k | 评估阶段对接 `MatterSim`/`CGCNN` 能量打分后：写 `formation_energy.jsonl`（结构ID + 形成能/relax 能量 + stable/unstable）；并按能量重排序导出 top‑k CIF/npz 子集，更新 `valid_structure_rate` 与 `stability_rate` 的统计口径。 |
| P2 | TODO | stable 子集 Fine-tune | 基于 `formation_energy.jsonl`/`eval_tag` 选出 stable 样本，生成一个可复现的 fine-tune 数据清单（npz index list 或新 split json）；提供 `train_tokens.py` 的 fine-tune 模式（较小 LR、冻结部分层、可选只训练 tail/条件层）并记录与 base checkpoint 的对比指标。 |
| P2 | TODO | 生成→筛选→再训练 自回流闭环 | 新增脚本（如 `twodgen/evaluate/self_train_loop.py`）：固定 checkpoint 采样 → 评估打标（valid/stable/symmetry）→ 汇总为增量数据集 → 触发再训练（或仅更新 loss 权重/条件层）；确保每轮产物（config/hash/metrics）可追踪。 |
| P1 | TODO | 把 dataset 的 `min_dist` 彻底标准化 | 当前有 `dataset.extra['min_dist']`、`prepare_c2db_tokens` 写入的 `min_dist`、以及 `eval_samples` 重新算的 `min_dist`；实现一个统一函数（例如 `twodgen/common/metrics.py::min_dist_exact`) 并在三处调用/对齐，减少“训练看似不碰撞但评估大量碰撞”的统计偏差。 |
| P1 | TODO | 新颖性/多样性评估指标 | 在 `evaluate/` 增加 `novelty/diversity`：对生成结构做去重（可用 CIF 标准化 hash 或简单指纹），输出 unique rate；再用结构指纹（可先做简化版：`(gram6, n_atoms, element histogram)`）计算样本间距离的均值/分位数，写入 `tier0_metrics.json` 扩展字段。 |
| P1 | TODO | Checkpoint 自动评估闭环 | 实现脚本（例如 `twodgen/evaluate/auto_eval_checkpoints.py`）：扫描 `outputs/checkpoints/*/atomdenoiser_best.pt`，对每个 checkpoint 运行固定采样+eval，汇总为一张 `eval_summary.jsonl`（含 config hash、关键指标），并可选生成“问题清单草稿”供更新 `twodgen/problem.md`。 |
| P2 | TODO | Tier‑1 能量/稳定性对接（MatterSim） | 以现有 `evaluate/mattersim_energy.py` 为入口，把 relax/energy 结果接到 `run_pipeline.py`，产出 `stability_rate`、`formation_energy`（若 ref energies 可用）等统计，并与 Tier‑0/1 指标一起落盘。 |
| P2 | TODO | 添加最小化回归测试：eval NaN/命名/IndexedDataset | 在 `tests/` 中添加：1) `eval_samples` 在无同元素样本时 `min_dist_same_elem` 输出为 `null`；2) 写 `per_sample.jsonl` 且遇到 `per_sanmple.jsonl` 会备份；3) `_compute_dataset_min_dist` 可处理 `IndexedDataset` 包装（不再因 `index` 堆叠报错）。 |
