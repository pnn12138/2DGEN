
=== 1. 项目背景与现有结构（请先自动探索仓库） ===
- 使用 Python + PyTorch 环境（conda/uv 已配置）。
- 配置系统：Hydra，配置目录为 conf/，计划结构大致为：
  - conf/data/          # 数据相关配置
  - conf/model/         # 模型结构和超参数
  - conf/trainer/       # 优化器 / 训练参数
  - conf/loss/          # 损失函数/指标
  - conf/logging/       # 日志、输出路径
  - conf/Jdft2d.yaml    # （入口配置）描述 matbench_jdft2d 预测任务
- 源码目录（你可以根据实际情况适当调整命名）：
  - src/p_task/         # Python 包根目录
  - 其中已经有/将会有：
    - data/             # dataset & datamodule
    - models/           # 模型实现（GNN 基线等）
    - train.py          # 训练入口（Hydra main）

请你自动阅读现有代码/配置，如果有不一致的地方，请在不破坏现有功能的前提下适配。

=== 2. 任务目标：实现 ALIGNN 基准（matbench_jdft2d） ===
目标：实现一个可复现、结构清晰的 ALIGNN 基准流程，用于
Matbench 的 “matbench_jdft2d” 任务（从晶体结构预测 exfoliation_en）。

要求：
1. **数据部分**
   1.1 使用 matminer/matbench 官方数据：
       - 优先通过 matminer.datasets 或 matbench.bench:
         - 数据集名：matbench_jdft2d
         - 列： structure (pymatgen.Structure), exfoliation_en (目标，meV/atom)
   1.2 在本地做数据缓存：
       - 如果已有 data/jdft2d_cache/ 则复用；
       - 否则创建：
         - data/jdft2d_cache/jdft2d_meta.csv
           - 列：sample_id, formula, exfoliation_en
         - data/jdft2d_cache/structures.pkl
           - list[Structure]，顺序与 sample_id 对齐
       - 如有必要，新建一个 Python 脚本：
         - src/p_task/data/prepare_jdft2d_cache.py
         - 提供 main()，可从命令行运行：python -m p_task.data.prepare_jdft2d_cache
   1.3 构建 PyTorch Dataset / DataModule：
       - 类名建议：Jdft2dDataset / Jdft2dDataModule
       - 能够：
         - 从 meta CSV + structures.pkl 中读取样本
         - 根据 Matbench 官方 folds 划分 train/val/test（使用 MatbenchBenchmark 获取）
         - 输出：
           - 图数据（供 ALIGNN 使用）
           - 标签 exfoliation_en（单位按原始数据）

2. **图与特征构建（ALIGNN 输入）**
   2.1 实现或调用对 Structure → (atom graph, line graph) 的构图逻辑：
       - 可选：
         A) 直接依赖官方 ALIGNN 库（pip install alignn, jarvis-tools, dgl 等）
            - 尽量复用 alignn 中的 neighbor graph / angle graph 构建。
         B) 如果官方库不可用，则在本项目中简化实现一个 ALIGNN 风格模型：
            - 原子图：节点=原子，边=邻居关系（基于截断半径）
            - line graph：节点=原始图的边（表示角），边=共享中心原子的成对边
   2.2 特征：
       - 节点：元素 one-hot 或 embedding、原子序数等；
       - 边：距离、可能的径向基展开；
       - 角：夹角 cos(theta) 等。
   2.3 把构图部分封装成独立模块，例如：
       - src/p_task/models/alignn_graph.py
       - 或 src/p_task/data/graphs_alignn.py
       - 要求：训练前可以预处理成 .pt 缓存（可选）。

3. **模型实现：ALIGNN 主干网络**
   3.1 在 src/p_task/models/ 下新增 ALIGNN 模型实现：
       - 文件：src/p_task/models/alignn_model.py
       - 类：ALIGNN (nn.Module 或 LightningModule)
       - 关键结构：
         - 对原子图做 message passing（类似 GCN/GAT）、得到节点特征；
         - 对 line graph 做 message passing，融入角度信息；
         - 原子级 pooling → 图级 embedding；
         - MLP 回归头输出一个标量（剥离能）。
   3.2 超参数：
       - 参考 ALIGNN 论文与常规设定：
         - num_layers ~ 4–8
         - hidden_dim ~ 128 或 256
         - dropout ~ 0.1–0.2
         - 激活 ReLU/GELU 等
   3.3 提供一个高层封装类，例如：
       - Jdft2dAlignnTask 或 AlignnRegressor
       - 内部包含：
         - 模型本体
         - 损失函数（MAE/MSE）
         - 指标计算（MAE, RMSE）

4. **Hydra 配置集成**
   4.1 在 conf/model/ 下新增 ALIGNN 的配置：
       - conf/model/alignn.yaml：
         - 包含：
           - 名称：alignn
           - hidden_dim, num_layers, dropout, cutoff 等
           - optimizer 配置（若未放到 trainer/）
   4.2 在 conf/Jdft2d.yaml 中：
       - 添加/修改字段，使得可以通过：
         - python -m p_task.train task=Jdft2d model=alignn
         来选择 ALIGNN 作为模型
       - 确保其它任务/模型不会被破坏（保持默认行为）。
   4.3 在 conf/trainer/ 和 conf/loss/ 中：
       - 如有必要，增加对对齐 MAE 指标的配置（monitor: val/mae）。

5. **训练入口与运行脚本**
   5.1 修改/创建训练入口：
       - 位置建议：src/p_task/train.py
       - 使用 Hydra main：
         - 根据 config.data 选择 Jdft2dDataModule
         - 根据 config.model 选择 ALIGNN 模型
         - 使用 config.trainer 设置学习率、epochs、batch_size 等
   5.2 新建一个方便使用的 bash 脚本：
       - scripts/run_alignn_jdft2d.sh
       - 内容（示例，可根据实际 config 调整）：
         - #!/usr/bin/env bash
         - set -e
         - python -m p_task.train \
             task=Jdft2d \
             model=alignn \
             trainer.max_epochs=200 \
             trainer.batch_size=32
       - 给脚本增加可执行权限（chmod +x）。
   5.3 训练过程中：
       - 记录训练/验证 MAE、RMSE；
       - 最好保存 best checkpoint（根据 val/mae 最小）。

6. **验证与最小可运行示例**
   6.1 请确保：
       - 使用 CPU 也能跑一个最小 demo（例如 2 个 epoch，batch_size 很小），
         只要能完整跑通即可。
   6.2 训练完成后：
       - 在 logs 或 output 目录中保存：
         - 配置（yaml）
         - 最佳模型 checkpoint
         - 一个简单的 metrics.json（包含 train/val/test MAE）。
   6.3 在仓库根目录的 README 或 docs/README_Jdft2d_ALIGNN.md 中：
       - 自动追加一节：
         - 简要说明如何运行 ALIGNN 基线；
         - 示例命令：bash scripts/run_alignn_jdft2d.sh；
         - 简要说明当前获得的 MAE 结果（哪怕是小 epoch 结果也可以）。

7. **实现风格要求**
   - 尽量复用已有基础设施（数据加载、日志、Hydra 配置）。
   - 避免大规模重构，保持 PR 级别改动，清晰可读。
   - 所有新增 Python 文件需带简短文档字符串和类型注解（至少函数/类签名）。
   - 对于容易踩坑的地方（如 dgl/alignn 依赖版本），在 README 的该节中写清注意事项。

请分步完成以上任务，优先保证代码可以在本地最小规模运行通过，并自动生成 scripts/run_alignn_jdft2d.sh 这个训练脚本。
EOF
)"