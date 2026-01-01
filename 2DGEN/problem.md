# 2DGEN 训练-采样问题清单（按严重程度排序）

## 高（会显著影响训练/评估结论）
1. 训练时使用 “预计算邻居图” 与扩散噪声状态不一致（尤其在高噪声步）  
   - `scrip/train_tokens.py` 在 `--use-precomputed-neighbors` 打开时，会把 `C2DBTokenNPZDataset` 缓存的 `nbr_idx/nbr_mask` 直接喂给模型。  
   - 但扩散训练实际输入是 `frac_t/cell_t`（噪声混合后的坐标/晶胞，见 `common/atom_diffusion.py`），此时真实 kNN 与 clean 结构的邻居往往不同。  
   - 结果：模型注意力的邻居集合在许多 t 上是“错的”，会削弱几何学习能力，典型现象包括 min_dist 碰撞率偏高、面内晶格分布学得偏保守/偏窄。  
   - 建议：默认训练关闭 `--use-precomputed-neighbors`；或只在低噪声区间使用；或缓存并按 t/噪声重算邻居（代价较大）。
2. 评估逻辑仍是 3D PBC，未对齐 slab 的 2D PBC  
   - `evaluate/eval_samples.py` 的 MIC 默认对 3 个维度都 wrap（`shifts = round(df)`），和训练/采样常用的 slab `pbc_mask=1,1,0` 不一致。  
   - 这会系统性污染 `min_dist`、`cross_vacuum`、连通性等指标的判定与统计。
3. A++ v3 canonical 邻居定义与训练时 `frac+lattice` 几何基底可能不一致  
   - `data/prepare_c2db_tokens.py` 的 `nbr_idx` 基于 canonical slab 坐标 `(u,v,z_norm,a_hat,b_hat,n,t)` 的近邻。  
   - `model/atom_transformer.py` 的距离/edge bias 由 `frac_mic_dist(frac, lattice)` 得到（来自扩散状态的 `frac_t` 与 `cell_t` 解码后的 `lattice`）。  
   - 即便不考虑扩散噪声，这两套“坐标系/距离定义”也不是同一个几何问题；若混用，会造成邻居索引与距离特征语义不一致。

## 中（会影响可用性/可复现性）
1. `main.py` 提示的训练/采样命令路径不正确  
   - 实际脚本位于 `2DGEN/scrip/`，当前提示容易误导新手无法直接运行。
2. 评估指标未与训练/采样日志打通  
   - `evaluate/eval_samples.py` 已实现 Tier‑0/1 指标，但未集成到训练/采样流程，缺少系统化记录。
3. 关键断言与邻居构建单测缺失  
   - `guide.md` 建议的 finiteness/索引边界断言与邻居‑mask 单测未实现，仅有 smoke test。

## 低（历史/边缘问题，但建议记录）
1. `cholesky6` 的 clamp 尺度单位错误会导致晶胞长度塌缩到 ~41Å（已修复）  
   - 现已在 `scrip/train_tokens.py` 与 `scrip/sample_tokens.py` 修正为“内部尺度（物理长度 / sqrt(g_scale)）”统计与 clamp。  
   - 若复现到旧 checkpoint/旧脚本，仍可能遇到同类问题。
