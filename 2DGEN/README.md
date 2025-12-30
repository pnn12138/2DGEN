# 2DGEN 子项目说明

## 内容概览
- **默认路线（token）**：`model/atom_transformer.py` + `model/atom_denoiser.py`，以 `(Z,F,g)` token 表示进行扩散训练与采样。
- **扩散与损失**：`common/atom_diffusion.py`（v-pred + 动态权重 + 可选 Flow-Matching）。
- **数据**：`data/prepare_c2db_tokens.py` 生成 token 缓存 npz；`data/c2db_dataset.py` 提供 `C2DBTokenNPZDataset` 与 `C2DBAtomDataset`。
- **测试/训练**：`scrip/test_tokens.py`/`scrip/train_tokens.py`/`scrip/sample_tokens.py` 为 token 路线。
- **Legacy 网格路线**：`model/model.py` + `model/denoiser.py` + `scrip/train.py`/`scrip/sample_and_export.py`。

## 数据预处理（Token 默认）
1. 准备原始 CSV：`data/C2DB/c2db_summary.csv`（已包含 CIF 文本）。
2. 生成 token 缓存 npz：
  ```bash
  uv run python 2DGEN/data/prepare_c2db_tokens.py \
    --csv data/C2DB/c2db_summary.csv \
    --out data/C2DB/ache/c2db_tokens.npz \
    --max-atoms 24 --g-scale 100
  ```
  - `--max-atoms`：最多保留的原子数（超出则跳过该行）。
  - `--g-scale`：Gram6 缩放（训练中会乘回去恢复晶胞）。
  - `--niggli-reduce`：对晶胞做 Niggli 规约（可选，较慢）。

## Token 版扩散（默认）
基于 `(Z, F, g)` token 表示的 Transformer 扩散模型（默认路线）：

训练：
```bash
uv run python 2DGEN/scrip/train_tokens.py \
  --npz data/C2DB/ache/c2db_tokens.npz \
  --epochs 100 --batch-size 256 --lr 1e-4 \
  --g-scale 100 --k-neighbors 32 \
  --cell-rep cholesky6
```
可选：
- `--bucket-batches`：按原子数分桶减少 padding。
- `--niggli-reduce`：CSV 直读时对晶胞做 Niggli 规约。
- `--cell-rep gram6`：切回 Gram6（默认）。
- `--cell-init iso`：Cholesky-6D 采样先验用各向同性 `y_iso`，自动估计尺度与 clamp（也可手动覆盖）。

采样与导出：
```bash
uv run python 2DGEN/scrip/sample_tokens.py \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/20251229_124406/atomdenoiser_best.pt \
  --num-samples 10 --steps 20 --method euler \
  --max-atoms 24 --g-scale 100 --npz data/C2DB/ache/c2db_tokens.npz \
  --out-dir outputs/samples_tokens
```
可选：
- `--neighbor-update-steps`：采样时每 N 步更新 kNN（默认 1）。
- `--reduce-lattice` / `--niggli-reduce`：采样后晶胞规约。
- `--cell-init iso`：与训练一致时可开启各向同性先验。

## 网格版扩散（Legacy）
旧版 3×24×W 网格路线，仅用于对比或回溯：
```bash
uv run python 2DGEN/scrip/train.py \
  --data data/C2DB/ache/c2db_grid.npz \
  --epochs 1 --batch-size 64 --lr 1e-4
```
- 自动选择 GPU/CPU；每个 epoch 保存 checkpoint 到 `outputs/checkpoints/`。
- 可选参数：`--num-workers`、`--log-interval`、`--max-steps`（限制总步数便于快速验证）。
- 训练时模型预测 3 通道网格（原子/分数坐标/晶格参数），不再使用 mask 通道；采样导出将用第三通道还原晶格，不会强制立方晶格。

## 快速自测
Token 路线：
```bash
uv run python 2DGEN/scrip/test_tokens.py
```

Legacy 网格路线：
```bash
uv run python 2DGEN/scrip/test.py
```
执行随机输入的前向与反向传播，打印损失和张量形状。

## Legacy 采样与导出
使用训练好的 checkpoint 生成样本并导出 CIF（默认使用第三通道中的晶格参数）：
```bash
uv run python 2DGEN/scrip/sample_and_export.py \
  --checkpoint /home/pnn/2dgen/outputs/checkpoints/c2dbdenoiser_best.pt \
  --num-samples 10 --steps 20 --method euler \
  --max-atoms 24 --torus-freqs 1 2 4 8 \
  --atomic-scale 100 --lattice-scale 10 --angle-scale 180 \
  --out-dir outputs/samples_cif
```
- `--checkpoint` 建议指向 `c2dbdenoiser_best.pt` 或 `c2dbdenoiser_last.pt`。
- `--num-samples`/`--steps` 控制采样数量与步数；`--method` 选 euler/heun。
- `--lattice-scale`/`--angle-scale` 需与预处理时保持一致。
