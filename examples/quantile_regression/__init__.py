"""
VineQuantileRegressor, a scikit-learn wrapper for quantile regression using torchvinecopulib
"""

import logging
import os
import platform
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, RegressorMixin

import torchvinecopulib as tvc

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
load_dotenv()
DIR_WORK = Path(os.getenv("DIR_WORK"))
DIR_DATA = DIR_WORK / "examples" / "quantile_regression" / "data"
DIR_DATA.mkdir(parents=True, exist_ok=True)
DIR_OUT = DIR_WORK / "examples" / "quantile_regression" / "out"
DIR_OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


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


class VineQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit‐learn wrapper for “quantile = α” via a fitted VineCopula.
    """

    def __init__(
        self,
        alpha: float,
        mtd_vine: str = "rvine",
        mtd_bidep: str = "kendall_tau",
        thresh_trunc: float = 0.01,
        # * above are to be tuned
        is_cop_scale: bool = False,
        num_step_grid: int = 128,
        num_obs_max: int = None,
        random_state: int = 42,
        device: str | None = DEVICE,
    ):
        self.alpha = alpha
        self.mtd_vine = mtd_vine
        self.mtd_bidep = mtd_bidep
        self.thresh_trunc = thresh_trunc
        self.is_cop_scale = is_cop_scale
        self.num_step_grid = num_step_grid
        self.num_obs_max = num_obs_max
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        """
        X is 2D and y is 1D.
        """
        n, p = X.shape
        # * move into one torch tensor [X | y]
        obs = torch.hstack([X, y.view(-1, 1)])
        # * fit the vine
        self._vine = tvc.VineCop(
            num_dim=p + 1,
            is_cop_scale=self.is_cop_scale,
            num_step_grid=self.num_step_grid,
        ).to(self.device, dtype=torch.float64)
        self._vine.fit(
            obs=obs,
            first_tree_vertex=tuple(range(p)),
            mtd_vine=self.mtd_vine,
            num_obs_max=self.num_obs_max,
            mtd_bidep=self.mtd_bidep,
            thresh_trunc=self.thresh_trunc,
            seed=self.random_state,
        )
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        X is 2D and y is 1D.
        """
        n, p = X.shape
        X = X.to(self.device, dtype=torch.float64)
        # * data on top lv (will auto cdf if is_cop_scale=True)
        dct_v_s_obs = {(v,): X[:, v].view(-1, 1) for v in range(p)}
        # ! data deep in the vine, should be cop_scale manually: here it's α for QR!
        dct_v_s_obs[self._vine.sample_order] = torch.full(
            size=(n, 1), fill_value=self.alpha, device=self.device, dtype=torch.float64
        )
        # * draw exactly n “samples”, using partial sampling order of length 1
        # * obeying u_y = α exactly
        return (
            self._vine.sample(
                num_sample=n,
                sample_order=self._vine.sample_order[:1],
                dct_v_s_obs=dct_v_s_obs,
            )[:, -1]
            .cpu()
            .numpy()
        )

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
