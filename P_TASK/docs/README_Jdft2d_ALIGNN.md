# ALIGNN 基线进度说明（matbench_jdft2d）

## 本次完成的内容
- 新增 `p_task.data.prepare_jdft2d_cache`，直接从官方 `https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz` 下载并缓存，生成 `jdft2d_meta.csv`（sample_id、formula、exfoliation_en）和 `structures.pkl`。
- 从官方 `matbench_v0.1_validation.json` 提取 `matbench_jdft2d` 官方 5 折划分，写入 `P_TASK/data/matbench_jdft2d_splits.json`，避免依赖旧版 matbench 包。
- 实现 ALIGNN 图构建与模型：
  - `p_task.data.graphs_alignn`：基于 pymatgen 结构构造原子图、线图（边对之间的夹角），提供批处理 `collate_graphs`。
  - `p_task.data.matbench_jdft2d`：Dataset/DataModule 读取缓存、应用官方折，按配置再切出少量验证集。
  - `p_task.models.alignn_model`：简化版 ALIGNN（节点/边/角 MLP + 均值池化）Lightning 封装。
- Hydra 集成：
  - 默认入口 `conf/Jdft2d.yaml` 切到 `data: matbench_jdft2d`、`model: alignn`，新增 `task.fold`。
  - 新增 `conf/data/matbench_jdft2d.yaml`、`conf/model/alignn.yaml`。
- 训练入口与脚本：
  - `p_task.train` 选择 ALIGNN 或 CGCNN（兼容旧逻辑）。
  - `scripts/run_alignn_jdft2d.sh`：示例命令（2 epoch smoke test）。

## 运行与验证
1) 在仓库根目录一次性装好依赖（已有虚拟环境时先激活，再执行）：
```bash
uv sync
uv pip install torch pytorch-lightning  # 选择 CPU/GPU 版本均可
```
2) 生成/复用缓存（可重复运行，不会覆盖已有文件）：
```bash
python P_TASK/src/p_task/data/prepare_jdft2d_cache.py
```
3) 最小训练示例（CPU 也可运行，默认 fold=0，可通过第一个参数覆盖）：
```bash
bash P_TASK/scripts/run_alignn_jdft2d.sh [fold]
```
脚本会使用 `uv run` 调用 `P_TASK/scripts/train_alignn_jdft2d.py`（Hydra 入口仍在 `src/p_task/train.py`），无需 `-m`。
4) 评估已有 checkpoint（传入 checkpoint 路径，可选 fold 参数，默认 0）：
```bash
bash P_TASK/scripts/eval_alignn_jdft2d.sh /path/to/checkpoint.ckpt [fold]
```
评估脚本同样通过 `uv run` 调用，直接输出 test 指标。

## 结果/状态
- 已完成数据缓存与官方折还原，ALIGNN 最小可跑通配置。
- GPU 短训（RTX 3060 Laptop，3 epoch，batch 16）结果：
  - 验证 MAE ~57.1
  - 测试 MAE ~39.73
- 需在更长 epoch / 更大模型上进一步收敛以逼近官方基线。
- 仍需根据实际硬件调整 batch_size/epochs/学习率，并可补充更丰富的特征（径向基、角度展开等）以逼近官方 ALIGNN 表现。

## 可能的后续改进
- 进一步替换为官方 EdgeGated/GatedGCN 块或直接复用 `alignn` 包，以完全对齐消息传递细节。
- 尝试 OneCycleLR + 更大的 batch（需合适 steps_per_epoch）并微调 dropout/层数，逼近官方收敛速度。
- 增加 metrics.json/最佳 checkpoint 的自动持久化，并在 README 中更新具体指标。
- 可选：为构图阶段增加 `.pt` 缓存以减少重复开销。 

## 当前问题（对齐官方 ALIGNN 示例）
- **架构仍较官方简化**：已加入 RBF/ABF、LayerNorm+残差、hidden_dim=256，但仍未复刻官方 DGL 版的 EdgeGatedGraphConv/GatedGCN 等组件，消息传递仍为简化 MLP，容量与 inductive bias 低于官方。
- **优化策略仍有差异**：使用 CosineAnnealingLR + AdamW，并对标签做标准化；官方默认 OneCycleLR，需要 steps_per_epoch 支持并可能带来更快初期收敛。
- **图构建实现不同**：仍为手写 torch 邻居/line graph，未使用官方 DGL 邻接构图与批处理；虽参数对齐（cutoff=8Å、24 邻居、RBF/ABF），但边排序/角覆盖细节可能与官方存在差异。
