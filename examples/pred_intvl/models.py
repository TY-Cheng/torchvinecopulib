import copy

import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch.utils.data import DataLoader


class EncoderRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
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
        # a small MLP encoder mapping ℝ⁸ → ℝ^{latent_dim}
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
            nn.Linear(head_dim, 1),
        )

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
        optim.zero_grad()  # * claer out grad from last step
        y_hat, _ = model(x)
        # if using MSELoss, we want y to be float and same shape
        loss = criterion(
            y_hat,
            y.float() if isinstance(criterion, nn.MSELoss) else y,
        )
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
        # if using MSELoss, we want y to be float and same shape
        total_loss += criterion(
            y_hat,
            y.float() if isinstance(criterion, nn.MSELoss) else y,
        ).item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_base_model(
    train_loader,
    val_loader,
    input_dim,
    p_drop=0.5,
    latent_dim=10,
    num_epoch=100,
    patience=5,
    device=None,
):
    model = EncoderRegressor(
        input_dim=input_dim, latent_dim=latent_dim, p_drop=p_drop
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_z, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        _, z = model(x)
        all_z.append(z.cpu())
        all_y.append(y.cpu().float())  # ensure shape [B,1]
    Z = torch.cat(all_z, dim=0)  # [N, latent_dim]
    Y = torch.cat(all_y, dim=0)  # [N, 1]
    return Z, Y


def train_ensemble(
    M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim,
    latent_dim,
    device,
    num_epochs,
    patience=5,
):
    ensemble = nn.ModuleList()
    for m in range(M):
        torch.manual_seed(m)
        model_m = train_base_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            latent_dim=latent_dim,
            p_drop=0.0,
            num_epoch=num_epochs,
            patience=patience,
            device=device,
        )
        ensemble.append(model_m.cpu())
    return ensemble.to(device)


@torch.no_grad()
def ensemble_pred_intvl(ensemble, x, device, alpha=0.05):
    preds = []
    for model in ensemble:
        model = model.to(device)
        y_hat, _ = model(x.to(device))  # [batch,1]
        preds.append(y_hat.cpu())
    P = torch.stack(preds, dim=0)  # [M, batch, 1]
    mean = P.mean(dim=0)  # [batch,1]
    lo = P.quantile(alpha / 2, dim=0)  # [batch,1]
    hi = P.quantile(1 - alpha / 2, dim=0)  # [batch,1]
    return (
        mean.cpu().flatten().numpy(),
        lo.cpu().flatten().numpy(),
        hi.cpu().flatten().numpy(),
    )


@torch.no_grad()
def mc_dropout_pred_intvl(model, x, T=100, device=None, alpha=0.05):
    """
    x:      Tensor [batch,1,28,28]
    returns mean, lower, upper: each [batch,1]
    """
    model.eval()
    x = x.to(device)

    # re-enable dropout layers
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # collect T predictions
    preds = []
    for _ in range(T):
        y_hat, _ = model(x)  # [batch,1]
        preds.append(y_hat.cpu())
    P = torch.stack(preds, dim=0)  # [T, batch, 1]

    mu = P.mean(dim=0)  # [batch,1]
    lower = P.quantile(alpha / 2, dim=0)  # [batch,1]
    upper = P.quantile(1 - alpha / 2, dim=0)  # [batch,1]
    return (
        mu.cpu().flatten().numpy(),
        lower.cpu().flatten().numpy(),
        upper.cpu().flatten().numpy(),
    )


@variational_estimator
class BayesianEncoderRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim=10,
        hidden_dim1=128,
        hidden_dim2=64,
        hidden_dim3=32,
        hidden_dim4=16,
        hidden_dim5=8,
        head_dim=16,
        num_outputs=1,
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
            BayesianLinear(head_dim, num_outputs),
        )

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
    input_dim,
    latent_dim,
    num_epochs,
    device=None,
    patience=5,
    lr=1e-3,
    weight_decay=0.91,
):
    """
    Trains one BayesianEncoderRegressor with ELBO-loss and early stopping
    on validation MSE. Returns the model with best val-MSE.
    """
    model = BayesianEncoderRegressor(
        input_dim=input_dim,
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
            # sample_elbo will call forward(x) → y
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


@torch.no_grad()
def bnn_pred_intvl(model, x, T=200, alpha=0.05, device=None):
    model.train()  # keep Bayesian layers sampling
    preds = []
    for _ in range(T):
        out = model(x.to(device))
        # out might be y, or (y,z), or (y,z,kl) — we only need the first thing
        y_hat = out if isinstance(out, torch.Tensor) else out[0]
        preds.append(y_hat.cpu())
    P = torch.stack(preds, 0)  # [T, batch, 1]
    mu = P.mean(dim=0)  # [batch,1]
    lower = P.quantile(alpha / 2, dim=0)
    upper = P.quantile(1 - alpha / 2, dim=0)
    return (
        mu.cpu().flatten().numpy(),
        lower.cpu().flatten().numpy(),
        upper.cpu().flatten().numpy(),
    )
