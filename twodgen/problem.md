# twodgen 训练-采样问题清单（按严重程度排序）

> 范围：`/home/pnn/2dgen/twodgen` 目录内代码与脚本（token 扩散路线为主）。

## 本次“低合格率”直接结论（基于 `/home/pnn/2dgen/eval`）

现象（Tier-0/1 评估）：
- `valid_rate_eval = 0.238`，失败原因几乎全部是 `collision`（`fail_reason_counts.collision = 1524 / 2000`）
- `min_dist` 分布显著偏低：mean≈`1.187Å`，p10≈`0.651Å`；评估阈值是 `--eval-min-dist 1.5`
- 2D slab 指标（thickness/vacuum/cross_vacuum）基本正常，说明主要问题集中在**原子-原子近距离重叠**而非 slab 真空/厚度。

一句话：当前低合格率的主因不是“元素不对/厚度不对/晶格不对”，而是采样出来的结构里大量原子对的 MIC 距离 < `1.5Å`，被评估脚本判定为 collision。

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
