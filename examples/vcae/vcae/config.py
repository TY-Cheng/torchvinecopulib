import os
from dataclasses import dataclass

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Training-related
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "logs/"
    batch_size: int = 512 if torch.cuda.is_available() else 64
    max_epochs: int = 10
    accelerator: str = DEVICE
    devices: int = 1
    num_workers: int = 1  # or min(15, os.cpu_count())

    # Data-related
    dims: tuple[int, ...] = (1, 28, 28)
    val_train_split: float = 0.1

    # Model-related
    hidden_size: int = 64
    latent_size: int = 10
    learning_rate: float = 2e-4
    vine_lambda: float = 0.0
    # use_mmd: bool = False
    # mmd_sigmas: list[float] = [1e-1, 1, 10]
    # mmd_lambda: float = 10.0

config_mnist = Config(
    max_epochs=10,
    dims=(1, 28, 28),
    hidden_size=64,
    latent_size=10,
)

config_svhn = Config(
    max_epochs=50,
    dims=(3, 32, 32),
    hidden_size=128,
    latent_size=32,
)
