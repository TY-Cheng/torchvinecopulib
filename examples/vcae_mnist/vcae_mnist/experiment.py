import copy
import random

import numpy as np
import pytorch_lightning as pl
import torch

from .config import DEVICE, Config
from .metrics import compute_score
from .model import LitMNISTAutoencoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


def run_experiment(seed: int, config: Config):
    set_seed(seed)

    # Instantiate the model
    model_initial = LitMNISTAutoencoder()

    # Set up trainer
    trainer_initial = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        logger=False,  # disables all loggers
        enable_progress_bar=False,  # disables tqdm
        enable_model_summary=False,  # disables model summary printout
    )

    # Train the base autoencoder
    trainer_initial.fit(model_initial)

    # Stay on DEVICE
    model_initial.to(DEVICE)

    # Learn vine
    model_initial.learn_vine(n_samples=5000)

    # Extract test data
    rep_initial, _, data_initial, _, samples_initial = model_initial.get_data(stage="test")

    # Deepcopy for refit
    model_refit = copy.deepcopy(model_initial)

    # Set up trainer for refitting
    trainer_refit = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        logger=False,  # disables all loggers
        enable_progress_bar=False,  # disables tqdm
        enable_model_summary=False,  # disables model summary printout
    )

    # Refit the model
    trainer_refit.fit(model_refit)

    # Stay on DEVICE
    model_refit.to(DEVICE)

    # Extract test data
    rep_refit, _, data_refit, _, samples_refit = model_refit.get_data(stage="test")

    loglik_initial = model_initial.vine.log_pdf(rep_initial).mean().item()
    loglik_refit = model_refit.vine.log_pdf(rep_refit).mean().item()

    sigmas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    score_initial = compute_score(data_initial, samples_initial, DEVICE, sigmas=sigmas)
    score_refit = compute_score(data_refit, samples_refit, DEVICE, sigmas=sigmas)

    return {
        "seed": seed,
        "loglik": loglik_initial,
        "loglik_refit": loglik_refit,
        "mmd": score_initial.mmd,
        "mmd_refit": score_refit.mmd,
        "fid": score_initial.fid,
        "fid_refit": score_refit.fid,
    }
