"""
Main script to evaluate multivariate predictive uncertainty models.
This script trains and evaluates various models on multiple datasets,
collects predictive samples, and computes various uncertainty metrics.
"""

import logging
import pickle
import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
from data_utils import (
    DIR_WORK,
    device,
    extract_XY,
    get_logger,
    load_cds_libor_lag,
    load_crypto_tech_loader,
    load_etf_tech_loader,
    _set_seed,
)
from metrics import (
    energy_score,
    mv_kendall_tau_deviation,
    pinball_loss,
    variogram_score,
    winkler_score,
    mv_chatterjee_xi_deviation,
)
from models import (
    collect_bnn_samples,
    collect_ensemble_samples,
    collect_mc_dropout_samples,
    collect_vine_samples,
    extract_features,
    train_base_model,
    train_bnn,
    train_ensemble,
    train_vine,
)

# --- Configuration ---
DATASETS = {
    "cds": load_cds_libor_lag,
    "etf": load_etf_tech_loader,
    "crypto": load_crypto_tech_loader,
}
LATENT_DIM = 13
# Evaluation parameters
ALPHA = 0.05  # for interval/quantile metrics
QUANTILE_LOW = 0.025  # lower quantile for pinball loss
QUANTILE_HIGH = 0.975  # upper quantile for pinball loss


# Output directory
DIR_OUT = DIR_WORK / "examples" / "pu_mv" / "out"
DIR_OUT.mkdir(parents=True, exist_ok=True)

# * ===== is_test =====
is_test = False
# * ===== is_test =====
if is_test:
    DIR_OUT = DIR_OUT / "tmp"
    NUM_SEEDS = 2
    N_ENSEMBLE = 4
    S_VINE = 4
else:
    # NOTE
    NUM_SEEDS = 100
    N_ENSEMBLE = 100
    S_VINE = 200
NUM_EPOCHS = 100
T_MC = 200
T_BNN = 200
lst_lr = [1e-2, 1e-3, 1e-4]
LR = lst_lr[1]
lst_weight_decay = [0.0, 1e-3, 1e-2]
WEIGHT_DECAY = lst_weight_decay[1]
lst_p_drop = [0.1, 0.3, 0.5]
P_DROP = lst_p_drop[1]
PATIENCE = 5


def evaluate_samples(y_true, y_samples, scaler=None):
    """
    Compute all metrics for a set of multivariate samples.
    Returns dict of metric_name -> float.
    """
    return {
        "winkler": winkler_score(y_true, y_samples, alpha=ALPHA, Y_scaler=scaler),
        "pinball_low": pinball_loss(y_true, y_samples, quantile=QUANTILE_LOW, Y_scaler=scaler),
        "pinball_high": pinball_loss(y_true, y_samples, quantile=QUANTILE_HIGH, Y_scaler=scaler),
        "pinball_median": pinball_loss(y_true, y_samples, quantile=0.5, Y_scaler=scaler),
        "energy": energy_score(y_true, y_samples, Y_scaler=scaler),
        "variogram": variogram_score(y_true, y_samples, gamma=0.5, Y_scaler=scaler),
        "kendall_dev": mv_kendall_tau_deviation(y_true, y_samples, Y_scaler=scaler),
        "chatterjee_dev": mv_chatterjee_xi_deviation(y_true, y_samples, Y_scaler=scaler),
    }


def main():
    for ds_name, seed in product(
        DATASETS.keys(),
        range(NUM_SEEDS),
    ):
        # Setup logging and output file
        file_log = DIR_OUT / f"Eval_{ds_name}_{seed}.log"
        logger = get_logger(file_log, name=f"Eval_{ds_name}_{seed}")
        output_file = DIR_OUT / f"Metrics_{ds_name}_{seed}.pkl"
        if output_file.exists():
            logger.warning(f"Output {output_file} exists, skipping.")
            continue
        else:
            logger.info(f"Creating output file {output_file}")

        logger.info(f"Starting evaluation for dataset={ds_name}, seed={seed}")
        # Set seeds
        _set_seed(seed)

        # Load data
        try:
            (
                train_loader,
                val_loader,
                test_loader,
                X_scaler,
                Y_scaler,
                idx_train,
                idx_val,
                idx_test,
            ) = DATASETS[ds_name](seed=seed)
            logger.info(
                f"Loaded dataset {ds_name} with {len(train_loader.dataset)} train samples, "
                f"{len(val_loader.dataset)} val samples, {len(test_loader.dataset)} test samples."
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {ds_name}: {e}")
            continue

        # * get dimensions
        input_dim = X_scaler.mean_.shape[0]
        output_dim = Y_scaler.mean_.shape[0]
        logger.info(f"Input dim: {input_dim}, Output dim: {output_dim}, Latent dim: {LATENT_DIM}")
        X_test, Y_test = extract_XY(loader=test_loader, device=device)
        results = {
            "seed": seed,
            "dataset": ds_name,
            "metrics": {},
        }

        # 1) Ensemble
        try:
            logger.info("Training ensemble...")
            mdl = train_ensemble(
                M=N_ENSEMBLE,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=LATENT_DIM,
                device=device,
                seed=seed,
                num_epochs=NUM_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
            )
            Y_pred = collect_ensemble_samples(
                ensemble=mdl,
                loader=test_loader,
                device=device,
            )
            logger.info(f"Collected {Y_pred.shape[0]} samples from ensemble.")
            results["metrics"]["ensemble"] = evaluate_samples(
                y_true=Y_test, y_samples=Y_pred, scaler=Y_scaler
            )
        except Exception as e:
            logger.error(f"Ensemble failed: {e}")
            results["metrics"]["ensemble"] = None

        # 2) MC Dropout
        try:
            logger.info("Training base model with MC Dropout...")
            mdl = train_base_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=LATENT_DIM,
                seed=seed,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                p_drop=P_DROP,
                num_epoch=NUM_EPOCHS,
                patience=PATIENCE,
                device=device,
            )
            Y_pred = collect_mc_dropout_samples(
                model=mdl,
                loader=test_loader,
                device=device,
                T=T_MC,
            )
            logger.info(f"Collected {Y_pred.shape[0]} samples from MC Dropout.")
            results["metrics"]["mc_dropout"] = evaluate_samples(
                y_true=Y_test, y_samples=Y_pred, scaler=Y_scaler
            )
        except Exception as e:
            logger.error(f"MC Dropout failed: {e}")
            results["metrics"]["mc_dropout"] = None

        # 3) Bayesian Neural Network
        try:
            logger.info("Training Bayesian Neural Network...")
            mdl = train_bnn(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=LATENT_DIM,
                seed=seed,
                num_epochs=NUM_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                device=device,
            )
            Y_pred = collect_bnn_samples(model=mdl, loader=test_loader, device=device, T=T_BNN)
            logger.info(f"Collected {Y_pred.shape[0]} samples from BNN.")
            results["metrics"]["bnn"] = evaluate_samples(
                y_true=Y_test, y_samples=Y_pred, scaler=Y_scaler
            )
        except Exception as e:
            logger.error(f"BNN failed: {e}")
            results["metrics"]["bnn"] = None

        # 4) Vine copula
        try:
            logger.info("Training Vine Copula Model...")
            # train base to extract features
            mdl_base = train_base_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=LATENT_DIM,
                seed=seed,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                p_drop=0.0,
                # NOTE
                num_epoch=NUM_EPOCHS + 100,
                patience=PATIENCE,
                device=device,
            )
            Z_train, Y_train = extract_features(model=mdl_base, loader=train_loader, device=device)
            Z_val, Y_val = extract_features(model=mdl_base, loader=val_loader, device=device)
            Z_test, _ = extract_features(model=mdl_base, loader=test_loader, device=device)
            mdl = train_vine(
                Z_train=torch.vstack([Z_train, Z_val]),
                Y_train=torch.vstack([Y_train, Y_val]),
                seed=seed,
                # !CPU!
                device="cpu",
            )
            Y_pred = collect_vine_samples(model=mdl, Z_test=Z_test, S=S_VINE)
            logger.info(f"Collected {Y_pred.shape[0]} samples from Vine Copula.")
            results["metrics"]["vine"] = evaluate_samples(
                # !CPU!
                y_true=Y_test.cpu(),
                y_samples=Y_pred,
                scaler=Y_scaler,
            )
        except Exception as e:
            logger.error(f"Vine Copula failed: {e}")
            results["metrics"]["vine"] = None

        # Save results
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved metrics to {output_file}")
        logger.info(f"Completed evaluation for dataset={ds_name}, seed={seed}")
        torch.cuda.empty_cache()
    logging.shutdown()


if __name__ == "__main__":
    main()
