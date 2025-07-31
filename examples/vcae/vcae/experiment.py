import random
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

from .config import DEVICE, Config
from .metrics import compute_score
from .model import LitAutoencoder, LitMNISTAutoencoder, LitSVHNAutoencoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


def run_experiment(
    seed: int, config: Config, vine_lambda: float = 1.0, dataset: str = "MNIST"
) -> dict[str, Union[float, int, str]]:
    # Set the seed for reproducibility
    set_seed(seed)
    config.seed = seed

    # Instantiate the model
    model_initial: LitAutoencoder
    if dataset == "MNIST":
        model_initial = LitMNISTAutoencoder(config)
    elif dataset == "SVHN":
        model_initial = LitSVHNAutoencoder(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

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
    rep_initial, _, data_initial, decoded_initial, samples_initial = model_initial.get_data(
        stage="test"
    )

    # Reset the seed for refitting to avoid data leakage
    set_seed(seed)

    # Create a new model with the same configuration but reset vine lambda
    config.vine_lambda = vine_lambda
    model_refit = model_initial.copy_with_config(config)

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
    rep_refit, _, data_refit, decoded_refit, samples_refit = model_refit.get_data(stage="test")

    assert model_initial.vine is not None
    assert model_refit.vine is not None
    loglik_initial = model_initial.vine.log_pdf(rep_initial).mean().item()
    loglik_refit = model_refit.vine.log_pdf(rep_refit).mean().item()

    mse_initial = torch.nn.functional.mse_loss(decoded_initial, data_initial).item()
    mse_refit = torch.nn.functional.mse_loss(decoded_refit, data_refit).item()

    sigmas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    score_initial = compute_score(data_initial, samples_initial, sigmas=sigmas)
    score_refit = compute_score(data_refit, samples_refit, sigmas=sigmas)

    return {
        "seed": seed,
        "dataset": dataset,
        "mse_initial": mse_initial,
        "mse_refit": mse_refit,
        "loglik_initial": loglik_initial,
        "loglik_refit": loglik_refit,
        "mmd_initial": score_initial.mmd,
        "mmd_refit": score_refit.mmd,
        "fid_initial": score_initial.fid,
        "fid_refit": score_refit.fid,
    }
