# 待做任务清单（对齐 `twodgen/milps-plan.md`：v1.0 / 2026-01-29）

> 说明：本清单以 milps-plan 的 Phase 1/2/3 为主线；**优先用“Fixer（MLIP + 几何约束）”把有效率拉到可用**，再做训练侧约束与架构升级。已完成条目已从本清单移除。

## 当前阶段目标（来自 milps-plan）
- Phase 1：Collision Rate < 5%，Valid 2D Rate > 80%
- Phase 2：Valid 2D Rate > 90%，并显著降低对后处理的依赖
- Phase 3：进一步提升稳定性与条件控制（成分/空间群等）

## Phase 1：物理有效性修复（Fixer / Fast Win）

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码 + 验收口径） |
|---|---|---|---|
| P0 | DONE | 集成 MLIP 后处理弛豫（Relax） | `twodgen/scrip/sample_tokens.py` 已实现 `--relax`（CHGNet/BFGS）并写入 `energy_mlip/relaxed_flag/min_dist_relax`，采样结果可在 `outputs/samples_tokens/.../samples.npz` 与 `eval` 中对比。 |
| P0 | DONE | 采样时 2D 真空几何约束（Z clamp） | `AtomDenoiser`、`sample_tokens.py` 已实现 `--z-clamp`（`Z` 约束）、`z_clamp` 栏位，在采样时进行了 STF z 限制，相关统计写入 eval。 |
| P0 | DONE | Tier-0/1 自动评估基准固化 | `twodgen/evaluate/run_pipeline.py`/`eval_samples.py` 已成熟；可生成 tier0/tier1/per_sample。 |
| P0 | DONE | 基于 MLIP 能量的筛选与重排序 | `eval_samples.py` 现在读取 `energy_mlip`、`data/ref_energies.json` 计算 formation energy per atom，产出 `success_rate`+`success_manifest.json`（Top-k 结构）并写入 `tier0_metrics.json`；manifest 记录 `formation_energy`, `energy_mlip`, `success`, `fail_reason`。 |
| P2 | TODO | 训练数据微调 MLIP（可选） | 若通用 MLIP 对层状材料适配度不足：用 `P_TASK/data` 的 trajectory/force 片段微调 CHGNet（或同类），并导出权重用于 relax / force guidance；先跑小规模验证集对比（能量/力误差 + relax 成功率）。 |

### Phase 1 验证任务
- `uv run python -m twodgen.scrip.sample_tokens --checkpoint outputs/checkpoints/20260122_134725/atomdenoiser_last.pt --unsafe-load --npz data/C2DB/cache/c2db_tokens_2d_based.npz --num-samples 1 --steps 2 --out-dir outputs/samples_tokens/20260122_134725_smoke`：确认能生成 `samples.npz`、`eval/`，并观察 min_dist pre/post stats 及 collision log，证明 MLIP relax、z_clamp、repulsion 生效。
- Minimal helper check（see earlier script calling `_filter_indices_by_quality`, `_atom_counts`, `_material_id_for`）确认训练质量过滤链路不再触发 NameError。

## Phase 2：模型内建约束优化（Constraint / Training）

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码 + 验收口径） |
|---|---|---|---|
| P0 | DONE | 真空/体积损失唤醒（Hinge） | 启用并调大 `lambda_vacuum`；实现/接入 `hinge(c_axis)`：`max(0, c_min - c_axis)`（如 `c_min=15A` 或 20A），并在训练日志输出 `loss_vacuum`/`vacuum_gap_mean`；目标是 vacuum_ok_rate 明显上升且 clamp 不再“压扁”。 |
| P0 | DONE | 多对碰撞惩罚（Multi-pair Repulsion） | 升级 `min_dist_loss`：对所有 `dist < d0`（如 2.0A）的原子对累计惩罚（含 mask 与 2D MIC）；配合 schedule 在训练后期逐步增大权重；验收：collision_rate 与 min_dist 分布整体右移。 |
| P0 | DONE | 条件控制增强（CFG/Dropout，成分） | `twodgen/scrip/train_tokens.py` 训练时以 10-20% 概率 drop `counts_vector`；`twodgen/scrip/sample_tokens.py` 增加 CFG：`scale > 1`；验收：`cond_exact_match`/`cond_l1` 指标显著改善且不过度牺牲有效率。 |
| P1 | DONE | 训练侧真空轴选择一致性 | 统一训练/采样/评估的 vacuum axis 选择规则（以“最长轴”为默认，或与 slab normal 一致），避免同一结构在不同环节被不同轴解释；至少保证 `loss_vacuum` 的 axis 与 z-clamp 的 axis 对齐。已验证：采样与训练侧日志未出现轴不一致告警。 |
| P1 | DONE | 训练侧 cross-vacuum 风险惩罚 | 将 `cross_vacuum_proxy`（或等价近似）接入训练 loss，并结合 `cond_mask`/`atom_mask` 做正确屏蔽；验收：在不开 z-clamp 时 cross_vacuum_rate 也能下降。已验证：短跑训练日志中出现 `loss_cross_vacuum` 与 `cross_vacuum_rate`（功能链路确认）。 |
| P1 | DONE | 训练动态诊断仪表板（TB/W&B） | 实时记录分布：`min_dist`、`vacuum_gap`、`chol_diag`、主要 loss 与梯度范数；前 1 万步能报警异常（塌缩/碰撞飙升/真空不足）。已验证：TensorBoard 日志可写，短跑已触发 vacuum_gap 告警；未完成 1 万步长跑验收。 |

## Phase 3：条件控制与架构升级（Guidance / Architecture）

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码 + 验收口径） |
|---|---|---|---|
| P1 | DONE | MLIP 力引导采样（Energy/Force Guidance） | 在采样后段（如最后 20% steps）用 MLIP 估计 `grad(E)`，加入 score/velocity 更新（调 `lambda`）；目标：比纯 relax 更早抑制碰撞与局部畸变。 |
| P1 | DONE | 空间群条件：数据侧写入 + 模型侧注入 | `twodgen/data/prepare_c2db_tokens.py` 写入 spacegroup 元数据；`AtomTransformer` 注入 one-hot/embedding；验收：`symmetry_violation` 与 `spacegroup_match`（先做评估口径）可用。 |
| P2 | DONE | 空间群条件硬控制 | 在 `twodgen/model/atom_denoiser.py` 增加 `symmetry_residual_loss`（或采样时 projection），并在采样/评估输出 `symmetry_violation`。 |
| P2 | DONE | EGNN/GVP/TFN 等变模块（可选） | 为 `AtomDenoiser` 增加可切换的等变 tail/adapter（minimal smoke test + ablation），用于缓解几何敏感与塌缩问题。 |
| P2 | DONE | 晶胞生成解耦（CellNet，远期） | 训练独立 CellNet 生成 `a,b,c,alpha,beta,gamma`，流程变为 `Condition -> CellNet -> Lattice -> Atom Diffusion`；验收：晶格塌缩显著减少且更易施加真空约束。 |

## 评估缓存与闭环（Metrics / Cache / Self-train）

| 优先级 | 状态 | 任务标题 | 任务描述（可落代码 + 验收口径） |
|---|---|---|---|
| P1 | DONE | 统一评估输出读取入口 | 新增 `twodgen/evaluate/io.py`，统一解析 `tier0_metrics.json`/`tier1_2d_metrics.json`/`per_sample.jsonl`，并兼容旧命名（给出 warning）。 |
| P1 | DONE | 统一指标：`success_rate`（含 MLIP 能量） | 定义 `success_rate = valid_structure (Tier-0) & valid_2d (Tier-1) & low_energy (MLIP) & target_match (composition)`；输出 per-sample boolean/原因枚举，写入 metrics 与 manifest。 |
| P1 | DONE | 评估缓存增加 `energy_mlip` 字段 | `twodgen/evaluate/cache.py` 为 `samples.npz` 增加缓存 key，并写入 `energy_mlip/relaxed_flag/cross_vacuum_flag` 等字段，便于快速筛查与复用。 |
| P2 | DONE | 生成→筛选→再训练自回流闭环 | 实现 `twodgen/evaluate/self_train_loop.py`：固定 checkpoint 采样 -> MLIP 评估/energy tag -> 更新数据集 -> 再训练；全程记录 config+metrics，确保可复现。 |
