# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 本次“低合格率”直接结论（基于 `/home/pnn/2dgen/eval`）

现象（Tier-0/1 评估）：
- `valid_rate_eval = 0.238`，失败原因几乎全部是 `collision`（`fail_reason_counts.collision = 1524 / 2000`）
- `min_dist` 分布显著偏低：mean≈`1.187Å`，p10≈`0.651Å`；评估阈值是 `--eval-min-dist 1.5`
- 2D slab 指标（thickness/vacuum/cross_vacuum）基本正常，说明主要问题集中在**原子-原子近距离重叠**而非 slab 真空/厚度。

一句话：当前低合格率的主因不是“元素不对/厚度不对/晶格不对”，而是采样出来的结构里大量原子对的 MIC 距离 < `1.5Å`，被评估脚本判定为 collision。

## 高（会显著影响训练/评估结论）
1. 采样合格率与评估合格率的“阈值域不一致”（容易误判模型好坏）
   - 现象：`twodgen/scrip/sample_tokens.py` 内置的快速合法性检查用的是 `--min-dist`（默认 `0.8Å`），而你评估用的是 `--eval-min-dist 1.5Å`；因此 sampling log 打印的 `valid_rate` 会系统性偏高。
   - 关键点：`--min-dist` **只用于打标签/过滤 CIF**，并不会改变生成过程（不会“把原子推开”）。
   - 相关文件：`twodgen/scrip/sample_tokens.py`（`--min-dist` 检查）、`twodgen/evaluate/eval_samples.py`（`--min-dist` 判 collision）

2. 采样阶段缺少“防碰撞机制”（collision 是主失败模式）
   - 现象：`AtomDenoiser.generate()` 的投影仅包含：
     - `frac = frac - floor(frac)`（wrap 到 [0,1)）
     - `clip_lattice()`（体积/条件数 clip）
     - `project_geometry` 时仅做 `uv_angle` 单位化与 `z_norm` clip  
     没有任何针对 `min_dist` 的投影/排斥/重采样。
   - 影响：当模型在训练早期或某些条件下学不到“短程排斥”，采样很容易出现大量近距离重叠；这与你的评估统计完全一致（`collision` 占绝对多数）。
   - 相关文件：`twodgen/model/atom_denoiser.py`（`_project_step/_project_geometry_step`）、`twodgen/common/crystal.py`（MIC 距离实现，可用于后处理/repulsion）

3. `--project-geometry` 的使用条件容易踩坑：若 checkpoint 未训练几何头，开启会注入随机几何信号并干扰 `frac` 生成
   - 机制：`--project-geometry` 会在采样中引入/更新 `uv_angle/z_norm/lattice_param/t`，并把它们作为输入喂给 `AtomTransformer.forward()`（`geom_atom_mlp/geom_cell_mlp/t_mlp` 会直接加到 token 上）。
   - 风险：如果训练时未开启 `--use-geometry-fields`（`train_tokens.py` 保存的 `geometry_config.use_geometry_fields=False`），这些 head 的权重基本等价于“未训练/随机初始化”，采样时开启会让模型在每一步都被随机几何通道扰动，常见结果就是更不稳定、更容易 collision。
   - 建议：只有在训练时显式启用 `--use-geometry-fields` 并确认 loss 包含 `loss_uv/loss_zn/loss_lat/(loss_t)` 时，采样才启用 `--project-geometry`；否则保持关闭，并用 `--project-each-step` 仅做 wrap/clip。
   - 相关文件：`twodgen/scrip/train_tokens.py`（`--use-geometry-fields` 与 checkpoint 的 `geometry_config`）、`twodgen/scrip/sample_tokens.py`（`--project-geometry`）、`twodgen/model/atom_transformer.py`（几何通道注入点）

4. 条件采样默认只取 cond_npz 的第 0 行（导致你实际上在“重复同一个条件”采 2000 次）
   - 机制：当 checkpoint 的 `cond_config.use_condition=True` 且未指定 `--cond-random/--cond-index/--cond-first` 时，`sample_tokens.py` 默认 `indices = 0`，并将其 broadcast 到全部样本。
   - 影响：你看到的 `n_atoms` 恒定为 10，`cond` 分布也很窄，很可能是在某个单一 composition 上评估；这会掩盖模型在其它条件上的表现，也容易被“某个特别难的条件”拖垮总体合格率。
   - 建议：评估模型整体能力时默认加 `--cond-random`；想复现实验再固定 `--cond-index`。
   - 相关文件：`twodgen/scrip/sample_tokens.py`（cond 索引逻辑与 `num_atoms_list` 推导）

5.（更根本但需要工程投入）序列化排序（`order_idx`）在近简并结构上不连续，可能诱发同元素原子“坐标坍缩/重叠”
   - 机制：token cache 用 `lexsort(z, z_key, u_key, v_key, original_idx)` 生成序列顺序；当同元素原子在 (u,v,z) 上非常接近时，微小扰动会导致顺序翻转。
   - 影响：基于序列的 Transformer 在训练/采样时可能学到“交换不变”的平均解，表现为同元素原子靠得过近甚至重叠（这也是 set 生成常见失败模式之一）。
   - 相关文件：`twodgen/data/preprocess.py`（`order_idx` 生成）、`twodgen/data/prepare_c2db_tokens.py`、`twodgen/data/c2db_dataset.py`（对齐输出）

## 修复进展（已落地）
- 采样/评估阈值对齐：`sample_tokens.py` 使用 `--eval-min-dist` 作为采样合法性与评估阈值；`--min-dist` 保留为弃用别名并写入 samples 元信息。
- 采样后处理防碰撞：`AtomDenoiser.generate()` 支持最小距离 repulsion（`--min-dist-project/--min-dist-iter/--min-dist-strength/--min-dist-cut`）。
- project-geometry 护栏：未训练几何头时启用 `--project-geometry` 将直接报错，避免随机扰动。
- 条件采样默认随机：当 checkpoint 需要条件且未显式指定策略时，默认 `--cond-random` 并记录元信息。
- 评估统计补强：输出 collision 的 `min_dist` 分布，并在阈值不一致时提示告警。
- order_idx 稳定性：加入微小、确定性的 tie-break 以减少近简并排序翻转。

## 推荐采样命令（对齐评估阈值 + 防碰撞后处理）
```bash
uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/<RUN_STAMP>/atomdenoiser_best.pt \
  --num-samples 2000 --steps 50 --method heun \
  --max-atoms 24 --g-scale 100 --npz data/C2DB/cache/c2db_tokens_2d_based.npz \
  --use-ema --coord-frame canon --pbc-mask 1,1,0 \
  --eval --eval-min-dist 1.5 \
  --min-dist-project --min-dist-iter 8 --min-dist-strength 0.03 \
  --out-dir outputs/samples_tokens
```
