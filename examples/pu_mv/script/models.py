import copy
import random

import torch
import torch.nn as nn
import torch.nn.init as init
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from data_utils import _set_seed
from torch.utils.data import DataLoader

import torchvinecopulib as tvc


# %%
# * EncoderRegressor: MLP encoder for regression tasks
class EncoderRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 10,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        hidden_dim3: int = 32,
        hidden_dim4: int = 16,
        hidden_dim5: int = 8,
        head_dim: int = 16,
        p_drop: float = 0.3,
    ):
        super().__init__()
        # * MLP encoder mapping ℝ⁸ → ℝ^{latent_dim}
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(hidden_dim2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(latent_dim, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.BatchNorm1d(hidden_dim4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.BatchNorm1d(hidden_dim5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            #
            nn.Linear(hidden_dim5, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),  # keeps latent coords in [–1,1]
        )
        # --- head: ℝ^{latent_dim} → ℝ (house value) ---
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop / 2),
            nn.Linear(head_dim, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [batch, 8]
        z = self.encoder(x)  # [batch, latent_dim]
        y_hat = self.head(z)  # [batch, num_outputs]
        return y_hat, z


def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()  # * clear out grad from last step
        y_hat, _ = model(x)
        # * if using MSELoss, we want y to be float and same shape
        loss = criterion(y_hat, y.float())
        loss.backward()  # * compute new grad
        optim.step()  # * update weights with fresh grad
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat, _ = model(x)
        # * if using MSELoss, we want y to be float and same shape
        total_loss += criterion(y_hat, y.float()).item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_base_model(
    train_loader,
    val_loader,
    input_dim,
    output_dim,
    latent_dim,
    seed: int = 42,
    device=None,
    num_epoch=100,
    lr=1e-3,
    weight_decay=0.91,
    patience=5,
    p_drop=0.5,
):
    _set_seed(seed)
    model = EncoderRegressor(
        input_dim=input_dim, output_dim=output_dim, latent_dim=latent_dim, p_drop=p_drop
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0
    for epoch in range(1, num_epoch + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait > patience:
                # print(f"Early stopping at epoch {epoch}")
                break
        # print(f"Epoch {epoch:2d} — train loss: {loss:.4f}")
    model.load_state_dict(best_state)
    return model.to(device)


# %%
# * Ensemble training and prediction collection
def train_ensemble(
    M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim,
    output_dim,
    latent_dim,
    device,
    seed: int,
    num_epochs,
    lr=1e-3,
    weight_decay=0.91,
    patience=5,
):
    """
    Trains M base models and returns an ensemble of them.
    Each model is trained with a different random seed.
    """
    ensemble = nn.ModuleList()
    for m in range(seed, seed + M):
        model_m = train_base_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            seed=m,
            lr=lr,
            weight_decay=weight_decay,
            p_drop=0.0,
            num_epoch=num_epochs,
            patience=patience,
            device=device,
        )
        ensemble.append(model_m.cpu())
    return ensemble.to(device)


@torch.no_grad()
def collect_ensemble_samples(
    ensemble: nn.ModuleList,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Collects predictions from an ensemble of models.
    ensemble: nn.ModuleList of models
    loader: DataLoader for the dataset to collect predictions from
    device: torch.device to run the models on
    Returns a tensor of shape [M, N, output_dim] where M is the number of models in the ensemble,
    N is the number of samples in the dataset, and output_dim is the output dimension of the models.
    """
    lst_preds = []
    for model in ensemble:
        model = model.to(device)
        preds = []
        for x, _ in loader:
            x = x.to(device)
            y_hat, _ = model(x)
            preds.append(y_hat.cpu())
        lst_preds.append(torch.cat(preds, dim=0))  # [N, output_dim]
    return torch.stack(lst_preds, dim=0)  # [M, N, output_dim]


# %%
# * MC Dropout training and prediction collection
@torch.no_grad()
def collect_mc_dropout_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 100,
) -> torch.Tensor:
    """Collects T predictions from a model with MC dropout.
    model: nn.Module with dropout layers
    loader: DataLoader for the dataset to collect predictions from
    device: torch.device to run the model on
    T: number of predictions to collect
    Returns a tensor of shape [T, N, output_dim] where T is the number of predictions,
    N is the number of samples in the dataset, and output_dim is the output dimension of the model.
    """
    model = model.to(device)
    model.train()  # re-enable dropout layers
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    lst_preds = []
    for _ in range(T):
        preds = []
        for x, _ in loader:
            x = x.to(device)
            y_hat, _ = model(x)
            preds.append(y_hat.cpu())
        lst_preds.append(torch.cat(preds, dim=0))  # [N, output_dim]
    return torch.stack(lst_preds, dim=0)  # [T, N, output_dim]


# %%
# * Bayesian Encoder Regressor with ELBO-loss
@variational_estimator
class BayesianEncoderRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim=10,
        hidden_dim1=128,
        hidden_dim2=64,
        hidden_dim3=32,
        hidden_dim4=16,
        hidden_dim5=8,
        head_dim=16,
    ):
        super().__init__()
        # -- Encoder: vector 2 latent z
        self._encoder = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim1),
            nn.LeakyReLU(inplace=True),
            BayesianLinear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(inplace=True),
            BayesianLinear(hidden_dim2, hidden_dim3),
            nn.LeakyReLU(inplace=True),
            BayesianLinear(hidden_dim3, hidden_dim4),
            nn.LeakyReLU(inplace=True),
            BayesianLinear(hidden_dim4, hidden_dim5),
            nn.LeakyReLU(inplace=True),
            BayesianLinear(hidden_dim5, latent_dim),
            nn.Tanh(),  # keeps latent coords in [–1,1]
        )
        # -- Head: z 2 y
        self._head = nn.Sequential(
            BayesianLinear(latent_dim, head_dim),
            nn.LeakyReLU(0.1, inplace=True),
            BayesianLinear(head_dim, output_dim),
        )
        self.apply(self._init_bayesian_weights)

    def _init_bayesian_weights(self, m):
        if isinstance(m, BayesianLinear):
            # posterior mean / scale
            init.kaiming_normal_(m.weight_mu)  # init the posterior mean
            init.constant_(m.weight_rho, -3.0)  # init the posterior rho
            if hasattr(m, "bias_mu") and m.bias_mu is not None:
                init.zeros_(m.bias_mu)
                init.constant_(m.bias_rho, -3.0)

    def forward(self, x):
        # x: [batch, input_dim]
        z = self._encoder(x)
        y = self._head(z)
        return y

    def encode(self, x):
        return self._encoder(x)


def train_bnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dim: int,
    latent_dim: int,
    seed: int,
    num_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 0.91,
    patience: int = 5,
    device: torch.device = None,
):
    """Trains one BayesianEncoderRegressor with ELBO-loss and early stopping on
    validation MSE.

    Returns the model with best val-MSE.
    """
    _set_seed(seed)
    model = BayesianEncoderRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            # * sample_elbo will call forward(x) → y
            loss = model.sample_elbo(
                inputs=x,
                labels=y,
                criterion=criterion,
                sample_nbr=3,
                complexity_cost_weight=1 / len(train_loader.dataset),
            )
            loss.backward()
            optim.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = []
                for _ in range(10):
                    preds.append(model(x))
                y_hat = torch.stack(preds, 0).mean(dim=0).to(device)
                val_losses.append(criterion(y_hat, y.float()).item() * x.size(0))
        val_loss = sum(val_losses) / len(val_loader.dataset)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model.to(device)


def collect_bnn_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 100,
) -> torch.Tensor:
    """Collects T predictions from a BNN.
    model: BayesianEncoderRegressor with ELBO-loss
    loader: DataLoader for the dataset to collect predictions from
    device: torch.device to run the model on
    T: number of predictions to collect
    Returns a tensor of shape [T, N, output_dim] where T is the number of predictions,
    N is the number of samples in the dataset, and output_dim is the output dimension of the model.
    """
    model = model.to(device)
    model.train()
    lst_preds = []
    for _ in range(T):
        preds = []
        for x, _ in loader:
            x = x.to(device)
            y_hat = model(x)
            preds.append(y_hat.cpu())
        lst_preds.append(torch.cat(preds, dim=0))  # [N, output_dim]
    return torch.stack(lst_preds, dim=0)  # [T, N, output_dim]


# %%
# * Collect features and predictions from models for vine training
@torch.no_grad()
def extract_features(model, loader, device):
    """
    From a data loader, record targets Y and
    extracts the final layer latent features Z from a base model
    """
    model.eval()
    all_z, all_y = [], []
    for x, y in loader:
        _, z = model(x.to(device))
        all_z.append(z.cpu())
        all_y.append(y.cpu().float())  # ensure shape [B,output_dim]
    Z = torch.cat(all_z, dim=0)  # [N, latent_dim]
    Y = torch.cat(all_y, dim=0)  # [N, output_dim]
    return Z, Y


def train_vine(Z_train, Y_train, seed=42, device="cpu"):
    _set_seed(seed)
    mtd_bidep = random.choice(["ferreira_tail_dep_coeff", "kendall_tau"])
    mtd_tll = random.choice(["linear", "constant"])
    num_step_grid = 128  # default number of steps for grid-based copula estimation
    # * build vine cop on Z and Y together
    ZY_train = torch.cat([Z_train, Y_train], dim=1)
    if ZY_train.shape[0] > 50000:
        # * if too many samples, use a subset
        torch.manual_seed(seed)
        ZY_train = ZY_train[torch.randperm(ZY_train.shape[0])[:50000], :]
        mtd_tll = random.choice(["linear", "quadratic"])
        num_step_grid = 256
    # * all except last Y column, for MST in first stage
    first_tree_vertex = tuple(range(Z_train.shape[1]))
    model_vine: tvc.VineCop = tvc.VineCop(
        num_dim=ZY_train.shape[1],
        is_cop_scale=False,
        num_step_grid=num_step_grid,
    ).to(device=device)
    model_vine.fit(
        obs=ZY_train,
        first_tree_vertex=first_tree_vertex,
        mtd_bidep=mtd_bidep,
        mtd_kde="tll",
        mtd_tll=mtd_tll,
    )
    return model_vine


def collect_vine_samples(
    model: tvc.VineCop,
    Z_test: torch.Tensor,
    S: int = 50,
) -> torch.Tensor:
    """Collects S samples from a vine copula model.
    model: trained VineCop model
    Z_test: tensor of shape [N, latent_dim] with test features
    S: number of samples to collect
    Returns a tensor of shape [S, N, output_dim] where S is the number of samples,
    N is the number of samples in Z_test, and output_dim is the output dimension of the model.
    """
    lst_seed = list(range(S))
    num_row, latent_dim = Z_test.shape
    sample_order = model.sample_order[:-latent_dim]
    dct_v_s_obs = {(_,): Z_test[:, [_]] for _ in range(Z_test.shape[1])}
    lst_preds = []
    for seed in lst_seed:
        lst_preds.append(
            model.sample(
                num_sample=num_row,
                seed=seed,
                is_sobol=True,  # use Sobol sequence for reproducibility
                sample_order=sample_order,
                dct_v_s_obs=dct_v_s_obs,
            )[:, latent_dim:]  # [N, output_dim]
        )
    return torch.stack(lst_preds, dim=0)  # [S, N, output_dim]
