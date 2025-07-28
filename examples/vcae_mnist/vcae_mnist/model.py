import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import torchvinecopulib as tvc

from .config import DEVICE, config


class LitMNISTAutoencoder(pl.LightningModule):
    def __init__(
        self,
        data_dir: str = config.data_dir,
        hidden_size: int = 64,
        learning_rate: float = 2e-4,
        use_vine: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.use_vine = use_vine
        self.vine = nn.Identity()  # Ensures registration

        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        flat_dim = channels * width * height
        self.mnist_test = None
        self.mnist_train = None
        self.mnist_val = None

        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Encoder: flatten → hidden → latent
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 10),
        ).to(DEVICE)

        # Decoder: latent → hidden → image
        self.decoder = nn.Sequential(
            nn.Linear(10, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, flat_dim),
            nn.Sigmoid(),  # Ensure output in [0,1] range
        ).to(DEVICE)

    def set_vine(self, vine):
        if not isinstance(vine, tvc.VineCop):
            raise ValueError("Vine must be of type tvc.VineCop for tvc.")
        self.vine = vine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        if not self.use_vine:
            return x_hat.view(-1, *self.dims)
        else:
            # also return the latent representation
            z = z.view(-1, 10)
            return x_hat.view(-1, *self.dims), z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x.to(DEVICE)
        if self.use_vine:
            x_hat, z = self(x)
            loss = F.mse_loss(x_hat, x) - self.vine.log_pdf(z).mean()
        else:
            x_hat = self(x)
            loss = F.mse_loss(x_hat, x)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x.to(DEVICE)
        if self.use_vine:
            x_hat, z = self(x)
            loss = F.mse_loss(x_hat, x) - self.vine.log_pdf(z).mean()
        else:
            x_hat = self(x)
            loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x.to(DEVICE)
        if self.use_vine:
            x_hat, z = self(x)
            loss = F.mse_loss(x_hat, x) - self.vine.log_pdf(z).mean()
        else:
            x_hat = self(x)
            loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        if self.mnist_train is None:
            self.setup(stage="fit")
        return DataLoader(
            self.mnist_train,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def val_dataloader(self):
        if self.mnist_val is None:
            self.setup(stage="fit")
        return DataLoader(
            self.mnist_val,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def test_dataloader(self):
        if self.mnist_test is None:
            self.setup(stage="test")
        return DataLoader(
            self.mnist_test,
            batch_size=config.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=config.num_workers,
        )

    def get_data(self, stage):
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
        representations = torch.cat(representations, dim=0)
        labels = torch.cat(labels, dim=0)
        data = torch.cat(data, dim=0).flatten(start_dim=1)
        decoded = torch.cat(decoded, dim=0)
        if self.use_vine:
            samples = torch.cat(samples, dim=0)

        return representations, labels, data, decoded, samples

    def learn_vine(self, n_samples=5000):
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


# # Instantiate the LitMNISTAutoencoder
# model = LitMNISTAutoencoder()

# # Instantiate a trainer with the specified configuration
# trainer = pl.Trainer(
#     accelerator=config.accelerator,
#     devices=config.devices,
#     max_epochs=config.max_epochs,
#     logger=CSVLogger(save_dir=config.save_dir),
# )

# # Train the model using the trainer
# trainer.fit(model)

# # Train the vine
# model.learn_vine(n_samples=5000)
# # # Read in the training metrics from the CSV file generated by the logger
# # metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

# # # Remove the "step" column, which is not needed for our analysis
# # del metrics["step"]

# # # Set the epoch column as the index, for easier plotting
# # metrics.set_index("epoch", inplace=True)

# # # Create a line plot of the training metrics using Seaborn
# # sns.relplot(data=metrics, kind="line")

# # Train the vine
# model.learn_vine(n_samples=5000)

# # Copy the model for refitting
# model_refit = copy.deepcopy(model)

# # Instantiate a new trainer
# trainer_refit = pl.Trainer(
#     accelerator=config.accelerator,
#     devices=config.devices,
#     max_epochs=config.max_epochs,
#     logger=CSVLogger(save_dir=config.save_dir),
# )

# # Refit the model
# trainer_refit.fit(model_refit)

# # Test the model
# representation, labels, data, decoded, samples = model.get_data(stage="test")
# representation_refit, labels_refit, data_refit, decoded_refit, samples_refit = model_refit.get_data(
#     stage="test"
# )

# sigmas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
# score_model = compute_score(data, samples, DEVICE, sigmas=sigmas)
# score_refit_model = compute_score(refit_data, refit_samples, DEVICE, sigmas=sigmas)
# loglik_model = model.vine.log_pdf(representation).mean()
# loglik_refit_model = model_refit.vine.log_pdf(representation_refit).mean()
# print("Log-likelihood (original vs refit):")
# print(
#     f"Log-likelihood: {loglik_model} vs {loglik_refit_model} => original is {loglik_model / loglik_refit_model} x worse"
# )
# print("Model scores (original vs refit):")
# print(
#     f"MMD: {score_model.mmd} vs {score_refit_model.mmd} => original is {score_model.mmd / score_refit_model.mmd} x worse"
# )
# print(
#     f"FID: {score_model.fid} vs {score_refit_model.fid} => original is {score_model.fid / score_refit_model.fid} x worse"
# )
