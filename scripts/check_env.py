import shutil
import subprocess
import sys

import _bootstrap  # noqa: F401


def main() -> None:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")

    try:
        import numpy as np

        print(f"numpy: {np.__version__}")
    except ModuleNotFoundError:
        print("numpy: not installed")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch cuda device: {torch.cuda.get_device_name(0)}")
    except ModuleNotFoundError:
        print("torch: not installed")

    nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,driver_version,cuda_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"nvidia-smi: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError, PermissionError) as exc:
        print(f"nvidia-smi: unavailable ({exc})")


if __name__ == "__main__":
    main()
