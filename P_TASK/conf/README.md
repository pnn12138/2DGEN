# Hydra 配置说明

`conf/` 目录用于集中管理 Hydra 配置，目标是在不同实验中复用「数据、模型、训练流程、日志回调」等模块，并通过入口文件（如 `Jdft2d.yaml`）的 `defaults` 列表自由组合。

## 目录结构
```
conf/
├── README.md                # 当前说明文档
├── Jdft2d.yaml              # Jdft2d 剥离能回归任务入口（组合 defaults）
├── callbacks/               # 训练期回调配置
│   └── early_stop.yaml      # 早停 / 模型摘要等配置
├── data/                    # 数据集、预处理、DataLoader 设置
│   └── jdft2d.yaml          # CGCNN 基准的数据配置
├── logging/                 # 日志与 checkpoint 策略
│   └── tensorboard.yaml     # TensorBoard + checkpoint 规范
├── loss/                    # 损失函数、指标、目标后处理
│   └── exfoliation_reg.yaml # 剥离能回归的 loss/metrics 套餐
├── model/                   # 模型结构及优化器/调度器
│   └── cgcnn.yaml           # CGCNN 结构和训练超参
└── trainer/                 # Lightning Trainer 运行参数
    └── finetune.yaml        # GPU/精度/epoch 等训练配置
```

## 目录职责
- `data/`：定义数据源路径、缓存位置、数据增强、DataLoader 超参；可扩展到更多数据集。
- `model/`：声明模型结构参数，并嵌入优化器、学习率调度器等默认值，方便在不同 Trainer 中直接引用。
- `trainer/`：控制 PyTorch Lightning 的训练循环（epochs、设备、梯度裁剪、日志频率等）。
- `loss/`：配置任务级损失函数、评估指标以及预测后处理（如反标准化）。
- `logging/`：统一设置输出目录、TensorBoard/W&B 等记录器、checkpoint 命名与监控指标。
- `callbacks/`：附加的训练期回调，如早停、模型摘要、最佳模型保存等，可按需扩展。

通过在入口文件 `Jdft2d.yaml` 的 `defaults` 中引用上述子配置，即可快速定义完整实验；后续若新增任务，只需复制入口文件并替换对应的子配置即可。
