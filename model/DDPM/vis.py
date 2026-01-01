import math
import torch
import matplotlib.pyplot as plt

def show_training_batch(loader, num_samples=16, cols=4):
    """
    从 DataLoader 里取一个 batch，可视化图像及其标签。
    loader: 训练或验证的 DataLoader
    num_samples: 展示多少张
    cols: 可视化时的列数
    """
    images, labels = next(iter(loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 数据集里做了 [-1,1] 归一化，需要拉回 [0,1] 方便显示
    images = images.mul(0.5).add(0.5)  # (x + 1) / 2

    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for idx, ax in enumerate(axes.flatten()):
        if idx < images.size(0):
            ax.imshow(images[idx].squeeze().cpu(), cmap="gray")
            ax.set_title(f"label: {labels[idx].item()}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    return fig
from data import get_dataloader
from vis import show_training_batch

loader = get_dataloader(train=True)
fig = show_training_batch(loader, num_samples=16, cols=4)
plt.show()  # 或者保存 fig.savefig("train_batch.png")