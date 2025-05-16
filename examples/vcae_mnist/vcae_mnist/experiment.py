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
    model = LitMNISTAutoencoder()

    # Set up trainer
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        logger=False,  # disables all loggers
        enable_progress_bar=False,  # disables tqdm
        enable_model_summary=False,  # disables model summary printout
    )

    # Train the base autoencoder
    trainer.fit(model)

    # Stay on DEVICE
    model.to(DEVICE)

    # Learn vine
    model.learn_vine(n_samples=5000)

    # Deepcopy for refit
    model_refit = copy.deepcopy(model)

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

    # Evaluation
    rep, _, data, _, samples = model.get_data(stage="test")
    rep_r, _, data_r, _, samples_r = model_refit.get_data(stage="test")

    loglik = model.vine.log_pdf(rep).mean().item()
    loglik_r = model_refit.vine.log_pdf(rep_r).mean().item()

    sigmas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    score = compute_score(data, samples, DEVICE, sigmas=sigmas)
    score_r = compute_score(data_r, samples_r, DEVICE, sigmas=sigmas)

    return {
        "seed": seed,
        "loglik": loglik,
        "loglik_refit": loglik_r,
        "mmd": score.mmd,
        "mmd_refit": score_r.mmd,
        "fid": score.fid,
        "fid_refit": score_r.fid,
    }
