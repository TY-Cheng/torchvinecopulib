import logging
import os
import platform
import random
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))
DIR_DATA = DIR_WORK / "examples" / "vcae_more" / "data"
DIR_OUT = DIR_WORK / "examples" / "vcae_more" / "out"


def get_logger(
    log_file: Path | str,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    fmt_console: str = "%(asctime)s - %(levelname)s - %(message)s",
    fmt_file: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    name: str | None = None,
) -> logging.Logger:
    """Create (or retrieve) a module‐level logger that writes INFO+ to console and
    WARNING+ to file.

    Args:
        log_file: path to the file where warnings+ should be logged.
        console_level: logging level for console handler.
        file_level: logging level for file handler.
        fmt_console: format string for console output.
        fmt_file: format string for file output.
        name: name for the logger; defaults to str(log_file).
        device: the device string to log (e.g. "cuda"/"cpu"); if None, auto‐detects.

    Returns:
        A configured logging.Logger instance.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger_name = name or str(log_file)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # * prevent duplicate handlers if called multiple times
    if not logger.handlers:
        # * file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(fmt_file))
        logger.addHandler(fh)
        # * console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter(fmt_console))
        logger.addHandler(ch)
        # * initial banner
        logger.info("--- Logger initialized ---")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Python: {sys.version.replace(chr(10), ' ')}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

    return logger


def load_mnist_data(
    batch_size: int = 128,
    test_size: float = 0.3,
    val_size: float = 0.0,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Load MNIST dataset and return DataLoader."""
    from torchvision.datasets import MNIST

    dataset = MNIST(
        root=DIR_DATA,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to [C=1, H=28, W=28]
                transforms.Lambda(lambda x: x.to(torch.float32)),  # Ensures float32 type
            ]
        ),
    )
    # Split indices
    random.seed(seed)
    torch.manual_seed(seed)
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=seed)
    if val_size > 0:
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size / (1 - test_size), random_state=seed
        )
    else:
        val_indices = []
    # Create DataLoaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def load_celeb_data(
    batch_size: int = 128,
    test_size: float = 0.3,
    val_size: float = 0.0,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Fetch CelebA dataset, split into train/val/test, and wrap in PyTorch DataLoaders."""

    # Load CelebA dataset
    from torchvision.datasets import CelebA

    dataset = CelebA(
        root=DIR_DATA,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize to 64x64
                transforms.ToTensor(),  # Converts to [C=3, H=64, W=64]
                transforms.Lambda(lambda x: x.to(torch.float32)),  # Ensures float32 type
            ]
        ),
    )

    # Split indices
    random.seed(seed)
    torch.manual_seed(seed)
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=seed)
    if val_size > 0:
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size / (1 - test_size), random_state=seed
        )
    else:
        val_indices = []

    # Create DataLoaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def extract_XY(loader, device):
    """Extract features and targets from a DataLoader."""
    X, Y = [], []
    for x, y in loader:
        X.append(x.to(device))
        Y.append(y.to(device))
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)


if __name__ == "__main__":
    log_file = DIR_OUT / "data_util.log"
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True)
    logger = get_logger(
        log_file=log_file,
        name="DataUtilLogger",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    logger.info("Starting data loading...")

    # * Load MNIST data
    train_loader, val_loader, test_loader = load_mnist_data()
    logger.info(f"MNIST Train loader size: {len(train_loader.sampler)}")
    logger.info(f"MNIST Validation loader size: {len(val_loader.sampler)}")
    logger.info(f"MNIST Test loader size: {len(test_loader.sampler)}")

    # * Load CelebA data
    celeb_train_loader, celeb_val_loader, celeb_test_loader = load_celeb_data()
    logger.info(f"CelebA Train loader size: {len(celeb_train_loader.sampler)}")
    logger.info(f"CelebA Validation loader size: {len(celeb_val_loader.sampler)}")
    logger.info(f"CelebA Test loader size: {len(celeb_test_loader.sampler)}")
