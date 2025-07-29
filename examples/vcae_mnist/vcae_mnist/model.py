import abc
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import torchvinecopulib as tvc

from .config import DEVICE, config
from .metrics import mmd


class LitAutoencoder(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        dims: tuple[int, ...],
        data_dir: str = config.data_dir,
        hidden_size: int = 64,
        latent_size: int = 10,
        learning_rate: float = 2e-4,
        use_vine: bool = False,
        use_mmd: bool = False,
        mmd_sigmas: list = [1e-1, 1, 10],
        mmd_lambda: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["use_vine"])
        self.flat_dim = int(torch.prod(torch.tensor(dims)))

        # Placeholder for data attributes
        self.data_test: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        # Placeholders for the  vine copula
        self.use_vine = use_vine
        self.vine: Optional[tvc.VineCop] = None

        # Call subclass-defined builders
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    @abc.abstractmethod
    def build_encoder(self) -> nn.Module:
        """Subclasses must return an nn.Module mapping x -> z"""
        raise NotImplementedError

    @abc.abstractmethod
    def build_decoder(self) -> nn.Module:
        """Subclasses must return an nn.Module mapping z -> x̂"""
        raise NotImplementedError

    def set_vine(self, vine: tvc.VineCop) -> None:
        if not isinstance(vine, tvc.VineCop):
            raise ValueError("Vine must be of type tvc.VineCop for tvc.")
        latent_size: int = self.hparams["latent_size"]
        if not vine.num_dim == latent_size:
            raise ValueError(
                f"Vine dimension {vine.num_dim} does not match latent size {latent_size}."
            )
        self.vine = vine
        self.add_module("vine", vine)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        latent_size: int = self.hparams["latent_size"]
        dims: tuple[int, ...] = self.hparams["dims"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_out = z.view(-1, latent_size) if self.use_vine else None
        return x_hat.view(-1, *dims), z_out

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_vine:
            x_hat, z = self(x)
            recon_loss = F.mse_loss(x_hat, x)
            if self.vine is None:
                raise ValueError("Vine must be set before computing the loss.")
            vine_loss = -self.vine.log_pdf(z).mean()
            loss = recon_loss + vine_loss
            use_mmd: bool = self.hparams["use_mmd"]
            if use_mmd:
                mmd_sigmas: list = self.hparams["mmd_sigmas"]
                mmd_lambda: float = self.hparams["mmd_lambda"]
                z_vine = self.vine.sample(x.shape[0])
                z_vine = torch.tensor(z_vine, dtype=z.dtype, device=x.device)
                x_vine = self.decoder(z_vine)
                mmd_loss = mmd(x, x_vine, sigmas=mmd_sigmas)
                loss += mmd_lambda * mmd_loss
        else:
            x_hat, _ = self(x)
            loss = F.mse_loss(x_hat, x)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step to compute loss on training data."""
        x, _ = batch
        x.to(DEVICE)
        loss = self.compute_loss(x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step to compute loss on validation data."""
        x, _ = batch
        x.to(DEVICE)
        loss = self.compute_loss(x)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step to compute loss on test data."""
        x, _ = batch
        x.to(DEVICE)
        loss = self.compute_loss(x)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns the optimizer for training."""
        learning_rate: float = self.hparams["learning_rate"]
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        if self.data_train is None:
            self.setup(stage="fit")
        assert self.data_train is not None
        return DataLoader(
            self.data_train,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        if self.data_val is None:
            self.setup(stage="fit")
        assert self.data_val is not None
        return DataLoader(
            self.data_val,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        if self.data_test is None:
            self.setup(stage="test")
        assert self.data_test is not None
        return DataLoader(
            self.data_test,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def get_data(
        self, stage: str = "fit"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if stage == "fit" or stage is None:
            data_loader = self.train_dataloader()
        elif stage == "test":
            data_loader = self.test_dataloader()
        representations = []
        decoded = []
        labels = []
        data = []
        samples = []
        encoder_device = next(self.encoder.parameters()).device
        decoder_device = next(self.decoder.parameters()).device
        for batch in data_loader:
            x, y = batch
            x = x.to(encoder_device)
            with torch.no_grad():
                z = self.encoder(x).to(decoder_device)
                x_hat = self.decoder(z)
                if self.use_vine:
                    if self.vine is None:
                        raise ValueError("Vine must be set before sampling with use_vine=True.")
                    sample = self.vine.sample(x.shape[0])
                    sample = self.decoder(
                        torch.tensor(sample, dtype=z.dtype, device=decoder_device)
                    )
            decoded.append(x_hat)
            representations.append(z)
            labels.append(y)
            data.append(x)
            if self.use_vine:
                samples.append(sample)

        # Concatenate into a single tensor
        representations_tensor = torch.cat(representations, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        data_tensor = torch.cat(data, dim=0).flatten(start_dim=1)
        decoded_tensor = torch.cat(decoded, dim=0)
        samples_tensor = torch.cat(samples, dim=0) if self.use_vine else None

        return representations_tensor, labels_tensor, data_tensor, decoded_tensor, samples_tensor

    def learn_vine(self, n_samples: int = 5000) -> None:
        self.setup(stage="fit")

        representations, _, _, _, _ = self.get_data(stage="fit")

        representations_subset = representations[
            torch.randperm(representations.shape[0])[:n_samples]
        ]
        vine_tvc = tvc.VineCop(
            num_dim=representations_subset.shape[1],
            is_cop_scale=False,
            num_step_grid=30,
        ).to(DEVICE)
        vine_tvc.fit(
            obs=representations_subset,
            mtd_kde="tll",
        )
        self.set_vine(vine_tvc)
        self.use_vine = True


class LitMNISTAutoencoder(LitAutoencoder):
    def __init__(
        self,
        data_dir: str = config.data_dir,
        hidden_size: int = 64,
        latent_size: int = 10,
        learning_rate: float = 2e-4,
        use_vine: bool = False,
        use_mmd: bool = False,
        mmd_sigmas: list = [1e-1, 1, 10],
        mmd_lambda: float = 10.0,
    ):
        super().__init__(
            dims=(1, 28, 28),
            data_dir=data_dir,
            hidden_size=hidden_size,
            latent_size=latent_size,
            learning_rate=learning_rate,
            use_vine=use_vine,
            use_mmd=use_mmd,
            mmd_sigmas=mmd_sigmas,
            mmd_lambda=mmd_lambda,
        )

        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        self.transform = transforms.Compose([transforms.ToTensor()])

    def build_encoder(self) -> nn.Module:
        # Encoder: flatten → hidden → latent
        latent_size: int = self.hparams["latent_size"]
        hidden_size: int = self.hparams["hidden_size"]
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size),
        ).to(DEVICE)

    def build_decoder(self) -> nn.Module:
        # Decoder: latent → hidden → image
        latent_size: int = self.hparams["latent_size"]
        hidden_size: int = self.hparams["hidden_size"]
        return nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.flat_dim),
            nn.Sigmoid(),  # Ensure output in [0,1] range
        ).to(DEVICE)

    def prepare_data(self) -> None:
        data_dir: str = self.hparams["data_dir"]
        MNIST(data_dir, train=True, download=True)
        MNIST(data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        data_dir: str = self.hparams["data_dir"]
        if stage == "fit" or stage is None:
            data_full = MNIST(data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(data_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.data_test = MNIST(data_dir, train=False, transform=self.transform)