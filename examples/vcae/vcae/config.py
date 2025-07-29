import os
from dataclasses import dataclass

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")


@dataclass
class Config:
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "logs/"
    batch_size: int = 512 if torch.cuda.is_available() else 64
    max_epochs: int = 50
    accelerator: str = DEVICE
    devices: int = 1
    num_workers: int = 1  # min(15, os.cpu_count())


config = Config()
