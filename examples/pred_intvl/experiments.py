import logging
import pickle
from itertools import product

import numpy as np
import torch
from data_utils import (
    DIR_WORK,
    device,
    extract_XY,
    get_logger,
    load_california_housing,
    load_news_popularity,
)
from models import (
    bnn_pred_intvl,
    ensemble_pred_intvl,
    extract_features,
    mc_dropout_pred_intvl,
    train_base_model,
    train_bnn,
    train_ensemble,
)
from vine_wrapper import train_vine, vine_pred_intvl

DATASETS = {
    "california": load_california_housing,
    "news": load_news_popularity,
}
ALPHA = 0.05
DIR_OUT = DIR_WORK / "examples" / "pred_intvl" / "out"
DIR_OUT.mkdir(parents=True, exist_ok=True)
LATENT_DIM = 10

# * ===== is_test =====
is_test = False
# * ===== is_test =====

if is_test:
    DIR_OUT = DIR_OUT / "tmp"
    SEEDS = list(range(3))
    N_ENSEMBLE = 4
    NUM_EPOCH = 5
else:
    SEEDS = list(range(1, 100))
    N_ENSEMBLE = 20
    NUM_EPOCH = 150


if __name__ == "__main__":
    for seed, ds_name in product(
        SEEDS,
        DATASETS.keys(),
    ):
        file = DIR_OUT / f"PredIntvl_{ds_name}_{seed}.pkl"
        file_log = DIR_OUT / f"PredIntvl_{ds_name}_{seed}.log"
        logger = get_logger(file_log, name=f"PredIntvl_{ds_name}_{seed}")
        if file.exists():
            logger.warning(f"File {file} already exists, skipping...")
            continue
        logger.info(f"Running {ds_name} with seed {seed}...")
        # * set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        # * load data
        try:
            train_loader, val_loader, test_loader, xsc, ysc = DATASETS[ds_name](
                seed_val=seed
            )
            X_test, Y_test = extract_XY(test_loader, device)
        except Exception as e:
            logger.error(f"Error loading dataset {ds_name}: {e}")
            continue
        input_dim = X_test.shape[1]
        dct_result = {
            "seed": seed,
            "dataset": ds_name,
            "alpha": ALPHA,
            "Y_test": Y_test.flatten().cpu().numpy(),
        }
        # * train / fit: get PI on test
        # ! ensemble
        try:
            logger.info("Training ensemble...")
            torch.manual_seed(seed)
            ensemble = train_ensemble(
                M=N_ENSEMBLE,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                latent_dim=LATENT_DIM,
                device=device,
                num_epochs=NUM_EPOCH,
            )
            dct_result["ensemble"] = ensemble_pred_intvl(
                ensemble, X_test, device, ALPHA
            )
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            dct_result["ensemble"] = (None, None, None)
        # ! base with mc dropout
        try:
            logger.info("Training base model with MC dropout...")
            torch.manual_seed(seed)
            model_mcdropout = train_base_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                p_drop=0.3,
                latent_dim=LATENT_DIM,
                num_epoch=NUM_EPOCH,
                device=device,
            )
            dct_result["mcdropout"] = mc_dropout_pred_intvl(
                model_mcdropout, X_test, T=200, device=device, alpha=ALPHA
            )
        except Exception as e:
            logger.error(f"Error training base model with MC dropout: {e}")
            dct_result["mcdropout"] = (None, None, None)
        # ! bnn
        try:
            logger.info("Training BNN...")
            torch.manual_seed(seed)
            model_bnn = train_bnn(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                latent_dim=LATENT_DIM,
                device=device,
                num_epochs=NUM_EPOCH,
            )
            dct_result["bnn"] = bnn_pred_intvl(
                model_bnn, X_test, T=200, alpha=ALPHA, device=device
            )
        except Exception as e:
            logger.error(f"Error training BNN: {e}")
            dct_result["bnn"] = (None, None, None)
        # ! vine (also extracting Y_train, Y_test)
        try:
            logger.info("Training vine...")
            torch.manual_seed(seed)
            model_base = train_base_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                p_drop=0.0,
                latent_dim=LATENT_DIM,
                device=device,
            )
            Z_train, Y_train = extract_features(model_base, train_loader, device)
            model_vine = train_vine(Z_train, Y_train, device=device)
            Z_test, Y_test = extract_features(model_base, test_loader, device)
            dct_result["vine"] = vine_pred_intvl(
                model_vine, Z_test, alpha=ALPHA, seed=seed, device=device
            )
        except Exception as e:
            logger.error(f"Error training vine: {e}")
            dct_result["vine"] = (None, None, None)
        with open(file, "wb") as f:
            pickle.dump(dct_result, f)

        logger.info(f"Results saved to {file}")
        logging.shutdown()
