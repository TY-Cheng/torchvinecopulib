import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_size = 0.2
val_size = 0.1
load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))
DIR_DATA = DIR_WORK / "examples" / "pu_mv" / "data"
DIR_DATA.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(
    log_file: Path | str,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    fmt_console: str = "%(asctime)s - %(levelname)s - %(message)s",
    fmt_file: str = "%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s",
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


def load_cds_libor_lag(
    file_cds_libor_lag: Path | str = DIR_DATA / "df_cds_libor_lag.pkl.gz",
    num_tgt_col: int = 41,
    batch_size: int = 128,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    _set_seed(seed)
    df = pd.read_pickle(file_cds_libor_lag)
    tgt_col = df.columns[:num_tgt_col]
    feature_col = df.columns[num_tgt_col:]
    num_row = df.shape[0]
    test_size = int(num_row * test_frac)
    val_size = int(num_row * val_frac)
    train_size = num_row - test_size - val_size
    # * chronological split
    df_train, df_val, df_test = (
        df.iloc[:train_size],
        df.iloc[train_size : (train_size + val_size)],
        df.iloc[(train_size + val_size) :],
    )
    Y_train = df_train[tgt_col].values
    X_train = df_train[feature_col].values
    Y_val = df_val[tgt_col].values
    X_val = df_val[feature_col].values
    Y_test = df_test[tgt_col].values
    X_test = df_test[feature_col].values
    # !
    val_frac = random.uniform(0.9, 1.0)
    Y_val = Y_val[: int(len(Y_val) * val_frac)]
    X_val = X_val[: int(len(X_val) * val_frac)]
    # !
    idx_train = df_train.index.values
    idx_val = df_val.index.values
    idx_test = df_test.index.values
    # * fit scalers on train only
    X_scaler = StandardScaler().fit(X_train)
    Y_scaler = StandardScaler().fit(Y_train)
    # * transform all splits
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    Y_train = Y_scaler.transform(Y_train)
    Y_val = Y_scaler.transform(Y_val)
    Y_test = Y_scaler.transform(Y_test)
    # * to torch tensors
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Xva = torch.from_numpy(X_val).float()
    Yva = torch.from_numpy(Y_val).float()
    Xte = torch.from_numpy(X_test).float()
    Yte = torch.from_numpy(Y_test).float()
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
    return (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    )


def load_etf_tech_loader(
    file_etf_tech: Path | str = DIR_DATA / "df_etf_tech.pkl.gz",
    num_tgt_col: int = 45,
    batch_size: int = 128,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    _set_seed(seed)
    df = pd.read_pickle(file_etf_tech)
    tgt_col = df.columns[:num_tgt_col]
    feature_col = df.columns[num_tgt_col:]
    num_row = df.shape[0]
    test_size = int(num_row * test_frac)
    val_size = int(num_row * val_frac)
    train_size = num_row - test_size - val_size
    # * chronological split
    df_train, df_val, df_test = (
        df.iloc[:train_size],
        df.iloc[train_size : (train_size + val_size)],
        df.iloc[(train_size + val_size) :],
    )
    Y_train = df_train[tgt_col].values
    X_train = df_train[feature_col].values
    Y_val = df_val[tgt_col].values
    X_val = df_val[feature_col].values
    Y_test = df_test[tgt_col].values
    X_test = df_test[feature_col].values
    # !
    val_frac = random.uniform(0.95, 1.0)
    Y_val = Y_val[: int(len(Y_val) * val_frac)]
    X_val = X_val[: int(len(X_val) * val_frac)]
    # !
    idx_train = df_train.index.values
    idx_val = df_val.index.values
    idx_test = df_test.index.values
    # * fit scalers on train only
    X_scaler = StandardScaler().fit(X_train)
    Y_scaler = StandardScaler().fit(Y_train)
    # * transform all splits
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    Y_train = Y_scaler.transform(Y_train)
    Y_val = Y_scaler.transform(Y_val)
    Y_test = Y_scaler.transform(Y_test)
    # * to torch tensors
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Xva = torch.from_numpy(X_val).float()
    Yva = torch.from_numpy(Y_val).float()
    Xte = torch.from_numpy(X_test).float()
    Yte = torch.from_numpy(Y_test).float()
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
    return (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    )


def load_crypto_tech_loader(
    file_crypto_tech: Path | str = DIR_DATA / "df_crypto_tech.pkl.gz",
    num_tgt_col: int = 9,
    batch_size: int = 128,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    _set_seed(seed)
    df = pd.read_pickle(file_crypto_tech)
    tgt_col = df.columns[:num_tgt_col]
    feature_col = df.columns[num_tgt_col:]
    num_row = df.shape[0]
    test_size = int(num_row * test_frac)
    val_size = int(num_row * val_frac)
    train_size = num_row - test_size - val_size
    # * chronological split
    df_train, df_val, df_test = (
        df.iloc[:train_size],
        df.iloc[train_size : (train_size + val_size)],
        df.iloc[(train_size + val_size) :],
    )
    Y_train = df_train[tgt_col].values
    X_train = df_train[feature_col].values
    Y_val = df_val[tgt_col].values
    X_val = df_val[feature_col].values
    Y_test = df_test[tgt_col].values
    X_test = df_test[feature_col].values
    # !
    val_frac = random.uniform(0.9, 1.0)
    Y_val = Y_val[: int(len(Y_val) * val_frac)]
    X_val = X_val[: int(len(X_val) * val_frac)]
    # !
    idx_train = df_train.index.values
    idx_val = df_val.index.values
    idx_test = df_test.index.values
    # * fit scalers on train only
    X_scaler = StandardScaler().fit(X_train)
    Y_scaler = StandardScaler().fit(Y_train)
    # * transform all splits
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    Y_train = Y_scaler.transform(Y_train)
    Y_val = Y_scaler.transform(Y_val)
    Y_test = Y_scaler.transform(Y_test)
    # * to torch tensors
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Xva = torch.from_numpy(X_val).float()
    Yva = torch.from_numpy(Y_val).float()
    Xte = torch.from_numpy(X_test).float()
    Yte = torch.from_numpy(Y_test).float()
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
    return (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    )


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
    # * load CDS Libor Lag dataset
    (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    ) = load_cds_libor_lag()
    X_train, Y_train = extract_XY(train_loader, device)
    X_val, Y_val = extract_XY(val_loader, device)
    X_test, Y_test = extract_XY(test_loader, device)
    print("\n\nCDS Libor Lag dataset loaded successfully.")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"idx_train: {idx_train[:2]} ... {idx_train[-2:]}")
    print(f"idx_val: {idx_val[:2]} ... {idx_val[-2:]}")
    print(f"idx_test: {idx_test[:2]} ... {idx_test[-2:]}")
    print(f"Device: {device}")
    print(f"Number of workers: {train_loader.num_workers}")
    print(f"Pin memory: {train_loader.pin_memory}")
    print(f"Batch size: {train_loader.batch_size}")
    # * load ETF Tech dataset
    (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    ) = load_etf_tech_loader()
    X_train, Y_train = extract_XY(train_loader, device)
    X_val, Y_val = extract_XY(val_loader, device)
    X_test, Y_test = extract_XY(test_loader, device)
    print("\n\nETF Tech dataset loaded successfully.")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"idx_train: {idx_train[:2]} ... {idx_train[-2:]}")
    print(f"idx_val: {idx_val[:2]} ... {idx_val[-2:]}")
    print(f"idx_test: {idx_test[:2]} ... {idx_test[-2:]}")
    print(f"Device: {device}")
    print(f"Number of workers: {train_loader.num_workers}")
    print(f"Pin memory: {train_loader.pin_memory}")
    print(f"Batch size: {train_loader.batch_size}")
    # * load Crypto Tech dataset
    (
        train_loader,
        val_loader,
        test_loader,
        X_scaler,
        Y_scaler,
        idx_train,
        idx_val,
        idx_test,
    ) = load_crypto_tech_loader()
    X_train, Y_train = extract_XY(train_loader, device)
    X_val, Y_val = extract_XY(val_loader, device)
    X_test, Y_test = extract_XY(test_loader, device)
    print("\n\nCrypto Tech dataset loaded successfully.")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"idx_train: {idx_train[:2]} ... {idx_train[-2:]}")
    print(f"idx_val: {idx_val[:2]} ... {idx_val[-2:]}")
    print(f"idx_test: {idx_test[:2]} ... {idx_test[-2:]}")
    print(f"Device: {device}")
    print(f"Number of workers: {train_loader.num_workers}")
    print(f"Pin memory: {train_loader.pin_memory}")
    print(f"Batch size: {train_loader.batch_size}")
