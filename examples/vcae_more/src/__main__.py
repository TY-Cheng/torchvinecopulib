import copy
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pickle
from . import DIR_WORK, DIR_OUT, load_config_grid
from .data_util import extract_XY, get_logger, load_mnist_data, load_svhn_data
from .metric import compute_fid, compute_mmd
from .vcae import VineCopAutoEncoder

logger_main = get_logger(log_file=DIR_OUT / "main.log", console_level="INFO", file_level="DEBUG")
logger_main.info("Starting VCAE sweep...")


def main(cfg: SimpleNamespace) -> None:
    file_log = (
        DIR_OUT
        / f"{cfg.dataset}_{cfg.mtd_kde}_{cfg.mtd_bidep}_{cfg.lambda_nll_vine}_{cfg.seed}.log"
    )
    file_pkl = (
        DIR_OUT
        / f"{cfg.dataset}_{cfg.mtd_kde}_{cfg.mtd_bidep}_{cfg.lambda_nll_vine}_{cfg.seed}.pkl"
    )
    if not file_log.parent.exists():
        file_log.parent.mkdir(parents=True)
    if file_log.exists() or file_pkl.exists():
        return
    logger = get_logger(
        log_file=file_log,
        name=file_log.stem,
        console_level="INFO",
        file_level="DEBUG",
    )
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    res = {
        "recon": dict(),
        "joint": dict(),
        "mtd_kde": cfg.mtd_kde,
        "mtd_bidep": cfg.mtd_bidep,
        "lambda_nll_vine": cfg.lambda_nll_vine,
        "seed": cfg.seed,
    }
    # * init model (shared at first)
    base_model = VineCopAutoEncoder(
        model_type=cfg.model_type,
        input_shape=cfg.input_shape,
        latent_dim=cfg.latent_dim,
        mtd_vine=cfg.mtd_vine,
        mtd_bidep=cfg.mtd_bidep,
        mtd_kde=cfg.mtd_kde,
        tau_thresh=cfg.tau_thresh,
        num_step_grid=cfg.num_step_grid,
        device=cfg.device,
    ).to(cfg.device)

    # ----------------------
    # * Phase 1: AE pretraining
    # ----------------------
    optimizer_ae = torch.optim.Adam(
        base_model.parameters(), lr=cfg.lr_ae, weight_decay=cfg.weight_decay
    )
    for epoch in range(cfg.num_epochs_ae):
        base_model.train()
        total_loss = 0.0
        for x, _ in cfg.train_loader:
            x = x.to(cfg.device).float()
            optimizer_ae.zero_grad()
            x_hat, _ = base_model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer_ae.step()
            total_loss += loss.item() * x.size(0)
        logger.debug(
            f"[{cfg.dataset.upper()}] AE Epoch {epoch + 1}/{cfg.num_epochs_ae} MSE (train): {total_loss / len(cfg.train_loader.dataset):.4f}"
        )

    # --------------------------------------------
    # * Clone model, Fit vine
    # --------------------------------------------
    model_recon = base_model  # continues MSE
    model_joint = copy.deepcopy(base_model)  # switch to joint loss

    # * Fit vine on encoded latent z (from model_joint)
    X_train, _ = extract_XY(cfg.train_loader, device=cfg.device)
    with torch.no_grad():
        Z_train = model_joint.encode(X_train)
    model_joint.fit_vine(Z_train)
    del X_train, Z_train  # ! free memory
    logger.debug(
        f"Vine fitted with "
        f"kde {cfg.mtd_kde}, bidep {cfg.mtd_bidep}, "
        f"tau_thresh {cfg.tau_thresh}, num_step_grid {cfg.num_step_grid}"
    )

    # --------------------------------------------
    # * Phase 2: joint loss
    # --------------------------------------------
    optim_joint = torch.optim.Adam(model_joint.parameters(), lr=cfg.lr_joint)

    for epoch in range(cfg.num_epochs_joint):
        model_joint.train()
        total_loss_joint = 0.0

        for x, _ in cfg.train_loader:
            x = x.to(cfg.device).float()

            # * Joint model
            optim_joint.zero_grad()
            loss_joint = model_joint.loss_joint(x, lambda_nll_vine=cfg.lambda_nll_vine)
            loss_joint.backward()
            optim_joint.step()
            total_loss_joint += loss_joint.item() * x.size(0)

        logger.debug(
            f"[{cfg.dataset.upper()}] Joint Epoch {epoch + 1}/{cfg.num_epochs_joint} | "
            # f"Recon Loss (train): {total_loss_recon / len(cfg.train_loader.dataset):.4f} | "
            f"Joint Loss (train): {total_loss_joint / len(cfg.train_loader.dataset):.4f}"
        )
    # --------------------------------------------
    # * Final eval
    # --------------------------------------------
    # * Fit vine on encoded latent z for each (recon and joint)
    X_train, _ = extract_XY(cfg.train_loader, device=cfg.device)
    with torch.no_grad():
        Z_train = model_joint.encode(X_train)
    model_joint.fit_vine(Z_train)
    with torch.no_grad():
        Z_train = model_recon.encode(X_train)
    model_recon.fit_vine(Z_train)
    del X_train, Z_train  # ! free memory

    # * Compute metrics
    x_test, _ = extract_XY(cfg.test_loader, device=cfg.device)
    with torch.no_grad():
        # * model whose loss function is recon (MSE) only
        model_recon.eval()
        x_hat = model_recon(x_test)[0]
        res["recon"]["mse"] = F.mse_loss(x_hat, x_test).item()
        res["recon"]["mmd"] = compute_mmd(real=x_test, fake=x_hat)
        res["recon"]["fid"] = compute_fid(real=x_test, fake=x_hat)
        res["recon"]["nll_vine"] = (
            model_recon.get_neglogpdf_vine(model_recon.encode(x_test)).mean().item()
        )
        # * model whose loss function is joint (MSE + NLL)
        model_joint.eval()
        x_hat = model_joint(x_test)[0]
        res["joint"]["mse"] = F.mse_loss(x_hat, x_test).item()
        res["joint"]["mmd"] = compute_mmd(real=x_test, fake=x_hat)
        res["joint"]["fid"] = compute_fid(real=x_test, fake=x_hat)
        res["joint"]["nll_vine"] = (
            model_joint.get_neglogpdf_vine(model_joint.encode(x_test)).mean().item()
        )
    logger.info(
        f"[{cfg.dataset.upper()}] Final recon-only metrics: "
        f"MSE: {res['recon']['mse']:.4f}, "
        f"MMD: {res['recon']['mmd']:.4f}, "
        f"FID: {res['recon']['fid']:.4f}, "
        f"NLL-Vine: {res['recon']['nll_vine']:.4f}"
    )
    logger.info(
        f"[{cfg.dataset.upper()}] Final joint metrics: "
        f"MSE: {res['joint']['mse']:.4f}, "
        f"MMD: {res['joint']['mmd']:.4f}, "
        f"FID: {res['joint']['fid']:.4f}, "
        f"NLL-Vine: {res['joint']['nll_vine']:.4f}"
    )
    # --------------------------------------------
    # * Save results
    # --------------------------------------------
    with open(file_pkl, "wb") as f:
        pickle.dump(obj=res, file=f)
    logger.info(f"Results saved to {file_pkl}")


if __name__ == "__main__":
    lst_cfg = load_config_grid(DIR_WORK / "examples" / "vcae_more" / "config.yaml")
    logger_main.info(f"Total configurations loaded: {len(lst_cfg)}")
    logger_main.info("Starting sweep over configurations...")
    for idx, cfg in enumerate(lst_cfg):
        if cfg.is_test and idx > 1:
            break  # ! for testing, only run first 2 configs
        logger_main.info(f"\nRunning configuration {idx + 1}/{len(lst_cfg)}: {cfg}")
        # * Copy config to avoid modifying original
        cfg_copy = copy.deepcopy(cfg)
        # * Load data
        if cfg_copy.dataset == "mnist":
            cfg_copy.train_loader, cfg_copy.val_loader, cfg_copy.test_loader = load_mnist_data()
            cfg_copy.input_shape = next(iter(cfg_copy.train_loader))[0].shape[1:]  # (C, H, W)
            cfg_copy.model_type = "mlp"
            cfg_copy.latent_dim = 10
            cfg_copy.num_epochs_ae = 30
            cfg_copy.num_epochs_joint = 5
            cfg_copy.batch_size = 256
        elif cfg_copy.dataset == "svhn":
            cfg_copy.train_loader, cfg_copy.val_loader, cfg_copy.test_loader = load_svhn_data()
            cfg_copy.input_shape = next(iter(cfg_copy.train_loader))[0].shape[1:]  # (C, H, W)
            cfg_copy.model_type = "conv"
            cfg_copy.latent_dim = 20
            cfg_copy.num_epochs_ae = 10
            cfg_copy.num_epochs_joint = 2
            cfg_copy.batch_size = 256
        else:
            raise ValueError(f"Unknown dataset: {cfg_copy.dataset}")
        cfg_copy.latent_dim = int(cfg_copy.latent_dim)
        cfg_copy.test_size = float(cfg_copy.test_size)
        cfg_copy.val_size = float(cfg_copy.val_size)
        cfg_copy.seed = int(cfg_copy.seed)
        cfg_copy.lr_ae = float(cfg_copy.lr_ae)
        cfg_copy.num_epochs_ae = int(cfg_copy.num_epochs_ae)
        cfg_copy.lr_joint = float(cfg_copy.lr_joint)
        cfg_copy.num_epochs_joint = int(cfg_copy.num_epochs_joint)
        cfg_copy.weight_decay = float(cfg_copy.weight_decay)
        cfg_copy.lambda_nll_vine = float(cfg_copy.lambda_nll_vine)
        cfg_copy.num_step_grid = int(cfg_copy.num_step_grid)
        cfg_copy.tau_thresh = float(cfg_copy.tau_thresh)
        print(f"\n\nRunning configuration: {cfg_copy}")
        # main(cfg_copy)  # ! run main with copied config
        try:
            main(cfg_copy)
        except Exception as e:
            logger_main.error(f"Error in configuration {idx + 1}: {e}")
            continue
        logger_main.info(f"Finished configuration {idx + 1}/{len(lst_cfg)}")
        print("-" * 80)
