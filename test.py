"""Quick CUDA availability checker for the project torch install."""

from __future__ import annotations

def main() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime env check
        print("未安装 torch:", exc)
        return

    print("torch 版本:", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("CUDA 可用:", cuda_ok)
    if not cuda_ok:
        return

    try:
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print("GPU 数量:", device_count)
        for idx, name in enumerate(device_names):
            print(f"  GPU {idx}: {name}")

        # 简单的 CUDA 张量运算
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        z = (x + y).sum()
        print("CUDA 运算结果:", float(z))
    except Exception as exc:  # pragma: no cover - runtime env check
        print("CUDA 初始化/运算失败:", exc)


if __name__ == "__main__":
    main()
