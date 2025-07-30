import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvinecopulib as tvc


def get_latent_from_loader(model, loader, device):
    """Extract latent representations from a DataLoader using the given model."""
    lst_Z = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            lst_Z.append(z.cpu())  # ! move to CPU to save memory
    return torch.cat(lst_Z, dim=0)


class VineCopAutoEncoder(nn.Module):
    def __init__(
        self,
        model_type: str = "mlp",
        input_shape=(1, 28, 28),
        hidden_size=64,
        latent_dim=10,
        has_vine=False,
        mtd_vine="rvine",
        mtd_bidep="chatterjee_xi",
        mtd_kde="tll",
        tau_thresh=0.01,
        num_step_grid=256,
        device="cpu",
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.device = device
        self.has_vine = has_vine
        self.latent_dim = latent_dim
        self.mtd_vine = mtd_vine
        self.mtd_bidep = mtd_bidep
        self.mtd_kde = mtd_kde
        self.tau_thresh = tau_thresh
        self.num_step_grid = num_step_grid
        self.vine = nn.Identity()  # * placeholder for vine model

        # * flatten input shape
        channels, width, height = input_shape
        flat_dim = channels * width * height
        self.input_shape = input_shape
        self.flat_dim = flat_dim

        if self.model_type == "mlp":
            # * MNIST
            # * encoder
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size // 2, latent_dim),
            )
            # * decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, flat_dim),
                nn.Sigmoid(),
            )

        elif self.model_type == "conv":
            # * svhn
            # * encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 4, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(4, 8, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            # ! infer encoder_out_dim and shape using dummy forward pass
            with torch.no_grad():
                # * encoder_out_dim is the output dimension after the last conv layer
                self.encoder_out_dim = self.encoder(torch.zeros(1, *input_shape)).shape[1]
                self.encoder_out_shape = (8, 8, 8)  # match output of convs
            # * latent representation
            self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)

            # * decoder
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, self.encoder_out_dim),
                nn.LeakyReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (self.encoder_out_shape)),
                nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(4, channels, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError("model_type must be 'mlp' or 'conv'")
        self.to(self.device)

    def forward(self, x):
        if self.model_type == "mlp":
            z = self.encoder(x)
            x_hat = self.decoder(z).view(-1, *self.input_shape)
        else:  # conv
            h = self.encoder(x)
            z = self.fc_mu(h)
            h_dec = self.decoder_fc(z)
            x_hat = self.decoder(h_dec)
        # assert x_hat.shape == x.shape, f"Shape mismatch: got {x_hat.shape}, expected {x.shape}"
        return x_hat, z

    def reconstruct(self, x):
        x_hat, _ = self.forward(x)
        return x_hat

    def encode(self, x):
        if self.model_type == "mlp":
            return self.encoder(x)
        else:
            return self.fc_mu(self.encoder(x))

    def decode(self, z):
        if self.model_type == "mlp":
            return self.decoder(z).view(-1, *self.input_shape)
        else:
            h_dec = self.decoder_fc(z)
            return self.decoder(h_dec)

    def fit_vine(
        self,
        z: torch.Tensor,
    ):
        """Fit a vine copula model to the latent representations z."""
        if z.ndim != 2:
            raise ValueError("Latent representation z must be a 2D tensor.")

        vine = tvc.VineCop(
            num_dim=z.shape[1],
            is_cop_scale=False,
            num_step_grid=self.num_step_grid,
        ).to(self.device)
        with torch.no_grad():
            vine.fit(
                obs=z,
                mtd_vine=self.mtd_vine,
                mtd_bidep=self.mtd_bidep,
                mtd_kde=self.mtd_kde,
                thresh_trunc=self.tau_thresh,
            )
        self.vine = vine
        self.has_vine = True

    def sample_from_vine(
        self,
        num_sample: int,
        seed: int = 42,
        is_sobol: bool = False,
    ) -> torch.Tensor:
        """Sample latent representations from vine and decode to data space."""
        if not self.has_vine or not isinstance(self.vine, tvc.VineCop):
            raise RuntimeError("Vine model has not been fitted.")
        z_sampled = self.vine.sample(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
        )
        return self.decode(
            torch.tensor(
                z_sampled,
                dtype=torch.float32,
                device=self.device,
            )
        )

    def get_neglogpdf_vine(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the negative log-likelihood of the latent representations z."""
        if not self.has_vine or not isinstance(self.vine, tvc.VineCop):
            raise RuntimeError("Vine model has not been fitted.")
        return -self.vine.log_pdf(z)

    def loss_joint(
        self, x: torch.Tensor, recon_loss_fn=F.mse_loss, lambda_nll_vine: float = 0.1
    ) -> torch.Tensor:
        """Compute the joint loss: reconstruction loss + lambda * NLL."""
        x_hat, z = self.forward(x)
        recon_loss = recon_loss_fn(x_hat, x)
        if self.has_vine and isinstance(self.vine, tvc.VineCop) and lambda_nll_vine > 0:
            nll_loss = self.get_neglogpdf_vine(z).mean()
            return recon_loss + lambda_nll_vine * nll_loss
        else:
            return recon_loss
