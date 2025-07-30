import abc
import copy
from dataclasses import asdict
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN

import torchvinecopulib as tvc

from .config import DEVICE, Config


class LitAutoencoder(pl.LightningModule, abc.ABC):
    def __init__(self, config: Config) -> None:
        """Initialize the autoencoder with the given configuration."""
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.flat_dim = int(torch.prod(torch.tensor(self.hparams["dims"])))

        # Placeholders for data attributes
        self.data_test: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        # Placeholder for the vine copula
        self.vine: Optional[tvc.VineCop] = None

        # Call subclass-defined builders
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.transform = self.build_transform()

    @property
    @abc.abstractmethod
    def dataset_cls(self) -> type:
        """Subclasses must return the dataset class (e.g., MNIST, SVHN)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dataset_kwargs(self) -> dict:
        """Subclasses must return a dictionary of keyword arguments for the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_encoder(self) -> nn.Module:
        """Subclasses must return an nn.Module mapping x -> z"""
        raise NotImplementedError

    @abc.abstractmethod
    def build_decoder(self) -> nn.Module:
        """Subclasses must return an nn.Module mapping z -> x̂"""
        raise NotImplementedError

    @abc.abstractmethod
    def build_transform(self) -> transforms.Compose:
        """Subclasses must return a torchvision transforms.Compose for data preprocessing."""
        raise NotImplementedError

    def copy_with_config(self, new_config: Config) -> "LitAutoencoder":
        """Create a copy of the model with a new configuration (for refit)."""
        new_model = self.__class__(new_config)
        new_model.encoder.load_state_dict(self.encoder.state_dict())
        new_model.decoder.load_state_dict(self.decoder.state_dict())
        if self.vine is not None:
            new_model.vine = copy.deepcopy(self.vine)
        return new_model

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
        return x_hat.view(-1, *dims), z.view(-1, latent_size)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, z = self(x)
        loss = F.mse_loss(x_hat, x)
        if self.hparams["vine_lambda"] > 0:
            if self.vine is None:
                raise ValueError("Vine must be set before computing the loss.")
            vine_loss = -self.vine.log_pdf(z).mean()
            vine_lambda: float = self.hparams["vine_lambda"]
            loss += vine_lambda * vine_loss
            # use_mmd: bool = self.hparams["use_mmd"]
            # if use_mmd:
            #     mmd_sigmas: list = self.hparams["mmd_sigmas"]
            #     mmd_lambda: float = self.hparams["mmd_lambda"]
            #     z_vine = self.vine.sample(x.shape[0])
            #     z_vine = torch.tensor(z_vine, dtype=z.dtype, device=x.device)
            #     x_vine = self.decoder(z_vine)
            #     mmd_loss = mmd(x, x_vine, sigmas=mmd_sigmas)
            #     loss += mmd_lambda * mmd_loss
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

    def prepare_data(self) -> None:
        """Download the dataset if not already present."""
        data_dir = self.hparams["data_dir"]
        self.dataset_cls(data_dir, download=True, **self.dataset_kwargs["train"])
        self.dataset_cls(data_dir, download=True, **self.dataset_kwargs["test"])

    def setup(self, stage=None) -> None:
        """Setup the datasets for training, validation, and testing."""
        data_dir = self.hparams["data_dir"]

        if stage in ("fit", None):
            data_full = self.dataset_cls(
                data_dir, transform=self.transform, **self.dataset_kwargs["train"]
            )
            n_total = len(data_full)
            n_val = int(self.hparams["val_train_split"] * n_total)
            n_train = n_total - n_val

            generator = torch.Generator().manual_seed(self.hparams["seed"])
            self.data_train, self.data_val = random_split(
                data_full, [n_train, n_val], generator=generator
            )

        if stage in ("test", None):
            self.data_test = self.dataset_cls(
                data_dir, transform=self.transform, **self.dataset_kwargs["test"]
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        if self.data_train is None:
            self.setup(stage="fit")
        assert self.data_train is not None
        batch_size: int = self.hparams["batch_size"]
        num_workers: int = self.hparams["num_workers"]
        return DataLoader(
            self.data_train,
            batch_size=batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        if self.data_val is None:
            self.setup(stage="fit")
        assert self.data_val is not None
        batch_size: int = self.hparams["batch_size"]
        num_workers: int = self.hparams["num_workers"]
        return DataLoader(
            self.data_val,
            batch_size=batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        if self.data_test is None:
            self.setup(stage="test")
        assert self.data_test is not None
        batch_size: int = self.hparams["batch_size"]
        num_workers: int = self.hparams["num_workers"]
        return DataLoader(
            self.data_test,
            batch_size=batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=num_workers,
        )

    def get_data(
        self, stage: str = "fit"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extracts representations, labels, data, decoded outputs, and samples (e.g., to compute metrics)."""
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
                if self.vine is not None:
                    sample = self.vine.sample(x.shape[0])
                    sample = self.decoder(
                        torch.tensor(sample, dtype=z.dtype, device=decoder_device)
                    )
            decoded.append(x_hat)
            representations.append(z)
            labels.append(y)
            data.append(x)
            if self.vine is not None:
                samples.append(sample)

        # Concatenate into a single tensor
        representations_tensor = torch.cat(representations, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        data_tensor = torch.cat(data, dim=0).flatten(start_dim=1).flatten(start_dim=1)
        decoded_tensor = torch.cat(decoded, dim=0).flatten(start_dim=1)
        samples_tensor = (
            torch.cat(samples, dim=0).flatten(start_dim=1) if self.vine is not None else None
        )

        return representations_tensor, labels_tensor, data_tensor, decoded_tensor, samples_tensor

    def learn_vine(self, n_samples: int = 5000) -> None:
        """Learn the vine copula from a subset of representations."""
        if self.data_train is None:
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


class LitMNISTAutoencoder(LitAutoencoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def build_transform(self) -> transforms.Compose:
        """Returns a torchvision transforms.Compose for MNIST preprocessing."""
        return transforms.Compose([transforms.ToTensor()])

    @property
    def dataset_cls(self) -> type:
        """Returns the dataset class for MNIST."""
        return MNIST

    @property
    def dataset_kwargs(self) -> dict:
        """Returns a dictionary of keyword arguments for the MNIST dataset."""
        return {
            "train": {"train": True},
            "test": {"train": False},
        }

    def build_encoder(self) -> nn.Module:
        """Returns a fully connected encoder for MNIST."""
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
        """Returns a fully connected decoder for MNIST."""
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


class LitSVHNAutoencoder(LitAutoencoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def build_transform(self) -> transforms.Compose:
        """Returns a torchvision transforms.Compose for SVHN preprocessing."""
        return transforms.Compose([transforms.ToTensor()])

    @property
    def dataset_cls(self) -> type:
        """Returns the dataset class for SVHN."""
        return SVHN

    @property
    def dataset_kwargs(self) -> dict:
        """Returns a dictionary of keyword arguments for the SVHN dataset."""
        return {
            "train": {"split": "train"},
            "test": {"split": "test"},
        }

    def build_encoder(self) -> nn.Module:
        """Returns a convolutional encoder for SVHN."""
        latent_size = self.hparams["latent_size"]
        hidden_size = self.hparams["hidden_size"]
        return nn.Sequential(
            nn.Conv2d(
                3, hidden_size // 4, kernel_size=4, stride=2, padding=1
            ),  # → [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(
                hidden_size // 4, hidden_size // 2, kernel_size=4, stride=2, padding=1
            ),  # → [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(
                hidden_size // 2, hidden_size, kernel_size=4, stride=2, padding=1
            ),  # → [B, 128, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_size * 4 * 4, latent_size),
        ).to(DEVICE)

    def build_decoder(self) -> nn.Module:
        """Returns a convolutional decoder for SVHN."""
        latent_size = self.hparams["latent_size"]
        hidden_size = self.hparams["hidden_size"]
        return nn.Sequential(
            nn.Linear(latent_size, hidden_size * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_size, 4, 4)),
            nn.ConvTranspose2d(
                hidden_size, hidden_size // 2, kernel_size=4, stride=2, padding=1
            ),  # → [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_size // 2, hidden_size // 4, kernel_size=4, stride=2, padding=1
            ),  # → [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_size // 4, 3, kernel_size=4, stride=2, padding=1
            ),  # → [B, 3, 32, 32]
            nn.Sigmoid(),  # For pixel values in [0,1]
        ).to(DEVICE)