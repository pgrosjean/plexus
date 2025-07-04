
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import anndata as ad
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from plexus.ml_baselines.pretrained_ts_model_neuroactive import get_zarr_info
from plexus.ssl_training.datasets import generate_simulation_2_datasets

class CNN1DAutoencoder(pl.LightningModule):
    def __init__(self, latent_dim: int = 768, input_length: int = 1200, lr: float = 1e-3):
        """
        1D CNN-based autoencoder for univariate time series.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of latent space.
        input_length : int
            Length of the input time series (must match expected size).
        lr : float
            Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),   # [B, 16, 600]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 300]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 150]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # [B, 128, 75]
            nn.ReLU()
        )

        self.encoder_flattened_dim = 128 * 150
        self.latent_mapper = nn.Linear(self.encoder_flattened_dim, latent_dim)

        # -------- Decoder --------
        self.decoder_input = nn.Linear(latent_dim, self.encoder_flattened_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, 150]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # [B, 32, 300]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),    # [B, 16, 600]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),     # [B, 1, 1200]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape [B, 1200].

        Returns
        -------
        z : torch.Tensor
            Latent representation [B, latent_dim]
        x_hat : torch.Tensor
            Reconstructed output [B, 1200]
        """
        x = x.unsqueeze(1)  # [B, 1200] → [B, 1, 1200]
        x_encoded = self.encoder(x)      # [B, 128, 75]
        z = self.latent_mapper(x_encoded.view(x.size(0), -1))  # [B, latent_dim]

        x_decoded = self.decoder_input(z).view(x.size(0), 128, 150)
        x_hat = self.decoder(x_decoded).squeeze(1)  # [B, 1200]
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.squeeze()
        z, x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.squeeze()
        z, x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

def main():
    model = CNN1DAutoencoder()
    # Load the datasets
    train_ds, val_ds = generate_simulation_2_datasets(1)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=24, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=24, num_workers=2, shuffle=False)
    wandb_logger = WandbLogger(project="Plexus_baselines")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=100, accelerator=device, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    # Running inference
    zarr_info = get_zarr_info("../../plexus_data_archive/processed_zarr_files/simulation/")
    all_embeddings = []
    well_ids = []
    fov_ids = []
    model.eval()
    with torch.no_grad():
        for zarr_root in zarr_info.values():
            for well in tqdm(list(zarr_root.keys())):
                for fov in list(zarr_root[well].keys()):
                    signal = torch.Tensor(np.array(zarr_root[well][fov]['signal']))
                    if len(signal.shape) == 1:
                        signal = signal.unsqueeze(0)
                    embeddings, x_hat = model(signal)
                    all_embeddings.append(embeddings.float().detach().cpu().numpy())
                    fov_ids.append([fov]*len(embeddings))
                    well_ids.append([well]*len(embeddings))
    all_embeddings = np.vstack(all_embeddings)
    well_ids = np.hstack(well_ids)
    fov_ids = np.hstack(fov_ids)
    metadata_df = pd.read_csv("../../plexus_data_archive/metadata_files/simulation_metadata.csv")
    metadata_df['paths'] = metadata_df['well_id'].astype('str') + '/' + metadata_df['fov_id'].astype('str')
    metadata_df = metadata_df.rename(columns={'phenotype_number': 'simulation_phenotype'})
    metadata_df['well_group'] = metadata_df['zarr_file'].astype('str') + '_' + metadata_df['well_id'].astype('str')
    metadata_map = {k: v for k, v in zip(metadata_df['well_id'].values, metadata_df['simulation_phenotype'].values)}
    pheno_labels = np.array([metadata_map[x] for x in well_ids])
    obs = pd.DataFrame({'well_id': well_ids, 'fov_id': fov_ids, 'simulation_phenotype': pheno_labels})
    obs['zarr_file'] = ['simulation_2']*len(obs)
    obs['paths'] = obs['well_id'].astype('str') + '/' + obs['fov_id'].astype('str')
    obs['well_group'] = obs['zarr_file'].astype('str') + '_' + obs['well_id'].astype('str')
    adata = ad.AnnData(X=all_embeddings, obs=obs)
    adata.write_h5ad("../../plexus_embeddings/simulations/1dcnn_embed_simulation_2.h5ad")


if __name__ == "__main__":
    main()
