import logging
import os
import platform
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

latent_dim = 10
num_epochs = 10
batch_size = 128
test_size = 0.2
val_size = 0.1
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(random_seed)
load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))


def get_logger(
    log_file: Path | str,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    fmt_console: str = "%(asctime)s - %(levelname)s - %(message)s",
    fmt_file: str = "%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s",
    name: str | None = None,
) -> logging.Logger:
    """
    Create (or retrieve) a module‐level logger that writes INFO+ to console and WARNING+ to file.

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


def load_california_housing(
    batch_size: int = 128,
    test_size: float = test_size,
    val_size: float = val_size,
    seed_val: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Fetch California Housing, split into train/val/test,
    scale (features & target) on train only, and wrap in PyTorch DataLoaders.
    Returns: train_loader, val_loader, test_loader, x_scaler, y_scaler
    """
    torch.manual_seed(seed_val)

    # * download & split
    data = fetch_california_housing()
    X, y = data.data, data.target[:, None]
    # ! test set untouched
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed_val
    )
    # ! carve off validation from trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed_val
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    # * transform all splits
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    # * to torch tensors
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(y_train).float()
    Xva = torch.from_numpy(X_val).float()
    Yva = torch.from_numpy(y_val).float()
    Xte = torch.from_numpy(X_test).float()
    Yte = torch.from_numpy(y_test).float()
    # * wrap into DataLoaders
    train_loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(Xva, Yva),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, Yte),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, x_scaler, y_scaler


def load_news_popularity(
    batch_size: int = 128,
    test_size: float = test_size,
    val_size: float = val_size,
    seed_val: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Online News Popularity regression (UCI).
    Splits into train/val/test, scales on train only, wraps in DataLoaders.
    """
    torch.manual_seed(seed_val)
    # * fetch & split
    X, y = fetch_openml(
        "OnlineNewsPopularity", version=1, return_X_y=True, as_frame=False
    )
    X = X[:, 1:]  # ! remove the first column (website)
    y = y.astype(float).reshape(-1, 1)
    # ! test set untouched
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed_val
    )
    # ! carve off validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed_val
    )
    # * standardize
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    # * to torch tensors
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(y_train).float()
    Xva = torch.from_numpy(X_val).float()
    Yva = torch.from_numpy(y_val).float()
    Xte = torch.from_numpy(X_test).float()
    Yte = torch.from_numpy(y_test).float()

    # * DataLoaders
    train_loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(Xva, Yva),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, Yte),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, x_scaler, y_scaler


@torch.no_grad()
def extract_XY(loader, device):
    Xs, Ys = [], []
    for x, y in loader:
        Xs.append(x.to(device))
        Ys.append(y.to(device))
    return torch.cat(Xs, 0).cpu(), torch.cat(Ys, 0).cpu()


if __name__ == "__main__":
    # quick test / demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, x_scaler, y_scaler = load_california_housing()
    X_train, Y_train = extract_XY(train_loader, device)
    X_test, Y_test = extract_XY(test_loader, device)
    print(f"Train  X: {X_train.shape},  Y: {Y_train.shape}")
    print(f"Test   X: {X_test.shape},   Y: {Y_test.shape}")
    # quick sanity check
    train_loader, test_loader, xscaler, yscaler = load_news_popularity(
        batch_size=128, test_size=0.2, seed=42
    )
    Xtr, Ytr = extract_XY(train_loader, device)
    Xte, Yte = extract_XY(test_loader, device)
    print(f"News‐pop train: X={Xtr.shape}, Y={Ytr.shape}")
    print(f"News‐pop  test: X={Xte.shape}, Y={Yte.shape}")
