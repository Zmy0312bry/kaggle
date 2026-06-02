from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch cuda runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"device: {idx} - {torch.cuda.get_device_name(idx)}")
        print(f"vram: {total_gb:.2f} GB")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"matmul ok: mean={y.mean().item():.6f}")


if __name__ == "__main__":
    main()

