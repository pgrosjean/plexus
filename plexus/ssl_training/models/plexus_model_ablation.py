##########
# Author: Parker Grosjean
##########

###########################################
###########################################
## Importing Dependencies
###########################################
###########################################

from typing import Tuple, Union
import numpy as np

# Utils
from plexus.ssl_training.utils.io_utils import patchify
from plexus.ssl_training.utils.scheduler import HyperParameterScheduler

# Modeling
import pytorch_lightning as pl

# importing torch related dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

###########################################
###########################################
#### Defining PyTorch Model
###########################################
###########################################

def get_1d_sincos_pos_embed(embed_dim,
    num_positions):
    """
    Generate 1D sinusoidal positional embeddings.

    Parameters
    ----------
    embed_dim : int
        The dimension of each positional embedding.
    num_positions : int
        The number of positions (or time steps) for which to generate embeddings.

    Returns
    -------
    pos_embed : np.ndarray
        A numpy array of shape (num_positions, embed_dim) containing the sinusoidal positional embeddings.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Generate a sequence of positions from 0 to num_positions - 1
    positions = np.arange(num_positions, dtype=np.float32)  # Shape: (num_positions,)

    # Compute the scaling factors (frequencies)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # Shape: (embed_dim // 2,)

    # Compute the sinusoidal embeddings
    out = np.einsum('p,d->pd', positions, omega)  # Shape: (num_positions, embed_dim // 2)
    emb_sin = np.sin(out)  # Shape: (num_positions, embed_dim // 2)
    emb_cos = np.cos(out)  # Shape: (num_positions, embed_dim // 2)

    # Concatenate sine and cosine embeddings
    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # Shape: (num_positions, embed_dim)

    return pos_embed


class NetworkMAEAblations(pl.LightningModule):
    def __init__(self,
                 lr_scheduler: HyperParameterScheduler,
                 time_window: int = 50,
                 num_patches: int = 50,
                 num_register_tokens: int = 5,
                 num_channels: int = 12,
                 mask_percentage: float = 0.75,
                 random_init: bool = False,
                 permutation_invariant: bool = False):
        super(NetworkMAEAblations, self).__init__()
        # Defining modules an values used for model construction
        self.lr_scheduler = lr_scheduler
        self.mask_percentage = mask_percentage
        self.num_patches = num_patches
        self.patch_size = time_window
        self.num_channels = num_channels
        self.permutation_invariant = permutation_invariant
        # Defining the embeddings
        assert time_window % 2 == 0, "Time window must be even integer"
        self.encoder_pos_embed = nn.Parameter(torch.rand(num_patches, 768), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.rand(num_patches, 768), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(self.encoder_pos_embed.shape[-1], self.encoder_pos_embed.shape[0])
        self.encoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        self.encoder_channel_tokens = nn.Parameter(torch.rand(num_channels, 768))
        self.decoder_channel_tokens = nn.Parameter(torch.rand(num_channels, 768))
        self.distance_vector = nn.Parameter(torch.rand(1, 768))

        # Defining the mask token
        self.mask_token = nn.Parameter(torch.rand(1, 768))

        # Defining the patch embedder and reverse patch embedder
        self.patch_embedder = nn.Linear(time_window, 768)
        self.rev_patch_embeder = nn.Linear(768, time_window)
        # Defining the decoder embedder used for matching encoder and decoder hidden dims
        self.decoder_embed = nn.Linear(768, 768, bias=True)
        self.channel_embed_projector = nn.Linear(768, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.apply(self._init_weights)  # Initializing the weights that are independent of transformer blocks

        # Defining the encoder and decoder blocks
        if random_init:
            # Setting up the encoder
            model_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            model_decoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            norm_encoder = nn.LayerNorm(768)
            self.encoder_blocks = nn.Sequential(*model_encoder.blocks, norm_encoder)
            norm_decoder = nn.LayerNorm(768)
            self.decoder_blocks = nn.Sequential(*model_decoder.blocks, norm_decoder)
            # Initializing the weights
            self.apply(self._init_weights)
        else:
            pretrained_model_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            pretrained_model_decoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            norm_encoder = nn.LayerNorm(768)
            self.encoder_blocks = nn.Sequential(*pretrained_model_encoder.blocks, norm_encoder)
            norm_decoder = nn.LayerNorm(768)
            self.decoder_blocks = nn.Sequential(*pretrained_model_decoder.blocks, norm_decoder)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # Loss function to ensure diversity among vectors
    def channel_embedding_diversity_loss(self, embeddings):
        """
        Compute a loss to ensure that embeddings do not collapse.
        We use a contrastive loss that penalizes embeddings that are too close to each other.
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)  # Shape: (batch_size, num_vectors, num_vectors)

        # We want to maximize distances between different vectors
        # Use a hinge loss to ensure a minimum distance between distinct vectors
        margin = 1.0
        diversity_penalty = F.relu(margin - pairwise_distances).mean()

        return diversity_penalty

    def create_mask(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a mask for the input tensor x.
        The mask is a binary tensor of the same shape as x, where 0 indicates the value is kept and 1 indicates the value is removed.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [N, L, D], sequence
        mask_ratio : float
            The ratio of the sequence to mask
        
        Returns
        -------
        torch.Tensor
            The binary mask tensor 0 is keep, 1 is remove
        torch.Tensor
            The indices to keep
        torch.Tensor
            The indices used to restore the original order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_keep, ids_restore

    def random_masking(self,
                       x: torch.Tensor,
                       ids_keep: torch.Tensor) -> torch.Tensor:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, L, D], sequence
        ids_keep : float
            The ratio of the sequence to keep
        
        Returns
        -------
        torch.Tensor
            The masked tensor
        """
        _, _, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def _create_channel_embedding(self,
                                  tokens: torch.Tensor,
                                  mask: torch.Tensor,
                                  is_decoder: bool) -> torch.Tensor:
        """
        This function takes the input tensor and creates a cell token by hashing the downsampled signal.
        This is used to ensure permutation invariance in the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch, num_cells, time].
        mask:
            The mask tensor of shape [batch, num_cells*num_tokens].
        is_decoder : bool
            Whether the function is being used in the decoder.
        
        Returns
        -------
        torch.Tensor
            The channel embeddings of shape [batch, num_cells, 768].
        """
        if self.num_channels == 1:
            if is_decoder:
                channel_tokens = self.decoder_channel_tokens  # shape: [1, 768]
            else:
                channel_tokens = self.encoder_channel_tokens
            channel_embeddings = channel_tokens.unsqueeze(0).unsqueeze(0).repeat(tokens.shape[0], 1, self.num_patches, 1)  # shape: [batch, 1, num_patches, 768]
            channel_embeddings = channel_embeddings.flatten(1, 2)  # shape: [batch, num_patches*1, 768]
            return channel_embeddings
        else:
            # Shape of tokens: [batch, num_cells*num_tokens, 768]
            # Shape of mask: [batch, num_cells*num_tokens]
            expanded_mask = mask.unsqueeze(-1).repeat(1, 1, 768)  # shape: [batch, num_cells*num_tokens, 768]
            mask_bool = mask.reshape(mask.shape[0], self.num_channels, self.num_patches)  # shape: [batch, num_cells, num_tokens]
            denominator = mask_bool.sum(dim=2)  # shape: [batch, num_cells]
            masked_tokens = tokens * expanded_mask  # shape: [batch, num_cells*num_tokens, 768]
            # mean pooling per cell
            masked_tokens = masked_tokens.reshape(tokens.shape[0], self.num_channels, self.num_patches, -1)  # shape: [batch, num_cells, num_tokens, 768]
            channel_embeddings = masked_tokens.mean(dim=2)  # shape: [batch, num_cells, 768]
            expanded_mask = expanded_mask.reshape(expanded_mask.shape[0], self.num_channels, self.num_patches, -1)  # shape: [batch, num_cells, num_tokens, 768]
            channel_embeddings = masked_tokens.sum(dim=2) # shape: [batch, num_cells, 768]
            if not torch.any(denominator == 0):
                channel_embeddings = channel_embeddings / denominator.unsqueeze(-1) # shape: [batch, num_cells, 768]
            if is_decoder:
                channel_tokens = self.decoder_channel_tokens
            else:
                channel_tokens = self.encoder_channel_tokens
            dist_vec = self.distance_vector # shape: [1, 768]
            dist_vec = dist_vec.unsqueeze(0)  # shape: [1, 1, 768]
            dist_to_random_vector = torch.cdist(dist_vec, channel_embeddings, p=2).squeeze()  # shape: [batch, num_cells]
            if self.num_channels == 1:
                dist_to_random_vector.unsqueeze(1)
            # argsorting
            ids_shuffle = torch.argsort(dist_to_random_vector, dim=-1)  # shape: [batch, num_cells]
            channel_tokens_expanded = channel_tokens.unsqueeze(0).repeat(channel_embeddings.shape[0], 1, 1)  # shape: [batch, num_cells, 768]
            channel_embeddings = torch.gather(channel_tokens_expanded, 1, ids_shuffle.unsqueeze(-1).repeat(1, 1, 768))  # shape: [batch, num_cells, 768]
            channel_embeddings = channel_embeddings.unsqueeze(2) # shape: [batch, num_cells, 1, 768]
            channel_embeddings = channel_embeddings.repeat(1, 1, self.num_patches, 1) # shape: [batch_size, num_cells, num_patches, 768]
            channel_embeddings = channel_embeddings.flatten(1, 2) # shape: [batch_size, num_cellss*num_patches, 768]
            return channel_embeddings

    def forward_encoder(self, x: torch.Tensor,
                        mask_ratio: float) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor,
                                            torch.Tensor]:
        """
        Forward pass for the encoder without CLS or register tokens.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch, num_cells, time].
        mask_ratio : float
            The ratio of the sequence to mask.

        Returns
        -------
        torch.Tensor
            The latent tensor of shape [batch_size, num_visible_tokens, 768].
        torch.Tensor
            The mask tensor of shape [batch, num_channels*num_patches].
        torch.Tensor
            The indices used to restore the original order.
        torch.Tensor
            The encoder channel embeddings of shape [batch_size, num_channels*num_patches, 768].
        """
        # Patchify and flatten
        patches = patchify(x, self.patch_size)  # [batch_size, num_channels, n_patches, patch_size]
        patches = patches.flatten(1, 2)  # [batch_size, num_channels * n_patches, patch_size]

        # Project to token space
        tokens = self.patch_embedder(patches)  # [batch_size, num_channels * n_patches, 768]

        # Positional embeddings
        pos_embeds = self.encoder_pos_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, n_patches, 768]
        pos_embeds = pos_embeds.repeat(tokens.shape[0], self.num_channels, 1, 1).flatten(1, 2)  # [batch_size, num_channels * n_patches, 768]

        # Generate mask
        mask, ids_keep, ids_restore = self.create_mask(tokens, mask_ratio)

        # Channel embeddings
        if self.permutation_invariant:
            channel_embeddings = self._create_channel_embedding(tokens, mask, is_decoder=False)  # [batch, num_channels * n_patches, 768]
        else:
            channel_embeddings = self.encoder_channel_tokens.unsqueeze(0).unsqueeze(2)  # [1, num_channels, 1, 768]
            channel_embeddings = channel_embeddings.repeat(tokens.shape[0], 1, self.num_patches, 1).flatten(1, 2)  # [batch_size, num_channels * n_patches, 768]

        # Add embeddings
        tokens = tokens + pos_embeds + channel_embeddings

        # Apply masking
        masked_tokens = self.random_masking(tokens, ids_keep)  # [batch_size, num_visible_tokens, 768]

        # Encode
        encoded = self.encoder_blocks(masked_tokens)  # [batch_size, num_visible_tokens, 768]

        return encoded, mask, ids_restore, channel_embeddings
    
    def forward_inference(self,
                          x: torch.Tensor) -> torch.Tensor:
        """
        This function is used to run the model in inference mode.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch, num_cells, time].
        
        Returns
        -------
        torch.Tensor
            The latent tensor of shape shape: [batch_size, 1+num_register_tokens+num_cells*num_tokens, 768]
            To index into the latent tensor without the registers or cls token run the following:
            latent[:, (1+model.num_register_token):, :]
        """
        patches = patchify(x, self.patch_size) # shape: [batch_size, num_channels, n_patches, patch_size]
        patches = patches.flatten(1, 2) # shape: [batch_size, num_channels*n_patches, patch_size]
        tokens = self.patch_embedder(patches) # shape: [batch_size, num_channels*n_patches, 768]
        # setting up positional embeddings
        pos_embeds = self.encoder_pos_embed.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, n_patches, 768]
        pos_embeds = pos_embeds.repeat(tokens.shape[0], self.num_channels, 1, 1).flatten(1, 2)  # shape: [batch_size, num_channels*n_patches, 768]
        # setting up channel embeddings
        
        if self.permutation_invariant:
            mask, _, _ = self.create_mask(tokens, 0)
            channel_embeddings = self._create_channel_embedding(tokens, mask, is_decoder=False)
        else:
            channel_embeddings = self.encoder_channel_tokens.unsqueeze(0) # shape: [1, num_channels, 768]
            channel_embeddings = channel_embeddings.unsqueeze(2) # shape: [1, num_channels, 1, 768]
            channel_embeddings = channel_embeddings.repeat(tokens.shape[0], 1, self.num_patches, 1) # shape: [batch_size, num_channels, num_patches, 768]
            channel_embeddings = channel_embeddings.flatten(1, 2) # shape: [batch_size, num_channels*num_patches, 768]
        # shape: [batch, num_channels*num_patches, 768]
        # adding positional embeddings
        tokens = tokens + pos_embeds  # shape: [batch_size, num_channels*n_patches, 768]
        # adding channel embeddings
        tokens = tokens + channel_embeddings  # shape: [batch_size, num_channels*n_patches, 768]
        # Adding register tokens
        x = self.encoder_blocks(tokens)
        return x, None, None
        
    def forward_decoder(self,
                    x: torch.Tensor,
                    mask: torch.Tensor,
                    ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder without CLS or register tokens.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch_size, num_visible_tokens, 768]
        mask : torch.Tensor
            The mask tensor of shape [batch, num_channels*num_patches] used for generating the channel embeddings.
        ids_restore : torch.Tensor
            The indices used to restore the original order of the tokens.

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, num_channels, num_patches, patch_size]
        torch.Tensor
            The decoder channel embeddings used.
        """
        x = self.decoder_embed(x)  # shape: [batch_size, num_visible_tokens, 768]

        # Fill in masked tokens
        num_masked = ids_restore.shape[1] - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)  # shape: [batch_size, num_masked, 768]

        # Append mask tokens and restore the original order
        x_ = torch.cat([x, mask_tokens], dim=1)  # shape: [batch_size, num_tokens, 768]
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # shape: [batch_size, num_channels*num_patches, 768]

        # Add decoder positional embeddings
        pos_embeds = self.decoder_pos_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, n_patches, 768]
        pos_embeds = pos_embeds.repeat(x.shape[0], self.num_channels, 1, 1).flatten(1, 2)  # [batch_size, num_channels*n_patches, 768]
        x = x + pos_embeds

        # Add decoder channel embeddings
        if self.permutation_invariant:
            decoder_channel_embeddings = self._create_channel_embedding(x, mask, is_decoder=True)
        else:
            channel_embeddings = self.decoder_channel_tokens.unsqueeze(0).unsqueeze(2)  # [1, num_channels, 1, 768]
            channel_embeddings = channel_embeddings.repeat(x.shape[0], 1, self.num_patches, 1)  # [batch_size, num_channels, num_patches, 768]
            decoder_channel_embeddings = channel_embeddings.flatten(1, 2)  # [batch_size, num_channels*num_patches, 768]

        x = x + decoder_channel_embeddings

        # Transformer decoding
        x = self.decoder_blocks(x)  # [batch_size, num_channels*n_patches, 768]
        x = self.rev_patch_embeder(x)  # [batch_size, num_channels*n_patches, patch_size]

        # Reshape into patch grid
        x = x.reshape(x.shape[0], self.num_channels, self.num_patches, -1)  # [batch_size, num_channels, num_patches, patch_size]

        return x, decoder_channel_embeddings
    
    def forward_loss(self,
                     signals: torch.Tensor,
                     pred: torch.Tensor,
                     mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the loss calculation.

        Parameters
        ----------
        signals : torch.Tensor
            The input tensor of shape [batch, num_cells, time].
        pred : torch.Tensor
            The predicted tensor of shape [batch, num_channels, num_patches, patch_size].
        mask : torch.Tensor
            The mask tensor of shape [batch, num_cells*num_patches].
        
        Returns
        -------
        torch.Tensor
            The loss tensor.
        torch.Tensor
            The reconstructed tensor.
        torch.Tensor
            The time mask tensor of shape [batch, num_channels, time].
        """
        # signal shape: [batch, num_cells, time]
        # pred shape: [batch, num_channels, num_patches, patch_size]
        # mask shape: [batch, num_channels*num_patches]

        # unflatten the mask
        time_mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)  # shape: [batch, num_channels*num_patches, patch_size]
        time_mask = time_mask.reshape(signals.shape[0], self.num_channels, self.num_patches, -1)  # shape: [batch, num_channels, num_patches, patch_size]
        time_mask = time_mask.flatten(2, 3) # shape: [batch, num_channels, time]

        # flatten the pred for loss calculation
        pred_time = pred.flatten(2, 3)  # shape: [batch, num_channels, time]

        pred_tokens = pred.flatten(1, 2)  # shape: [batch, num_channels*num_patches, patch_size]
        patchified_signals = patchify(signals, self.patch_size)  # shape: [batch, num_channels, num_patches, patch_size]
        targets = patchified_signals.flatten(1, 2)  # shape: [batch, num_channels*num_patches, patch_size]

        loss = (targets - pred_tokens) ** 2  # shape: [batch, num_channels*num_patches, patch_size]
        loss = loss.mean(dim=-1) # shape: [batch, num_channels*num_patches]

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, pred_time, time_mask

    def forward(self,
                x: torch.Tensor,
                inference: bool = False) -> Tuple[torch.Tensor,
                                                  Union[None, torch.Tensor],
                                                  Union[None, torch.Tensor],
                                                  Union[None, torch.Tensor]]:
        """
        Forward pass for the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape [batch, num_cells, time].
        inference : bool
            Whether to run the model in inference mode.
        
        Returns
        -------
        torch.Tensor
            The latent tensor of shape [batch, num_channels, num_patches, patch_size].
        Union[None, torch.Tensor]
            The loss tensor.
        Union[None, torch.Tensor]
            The reconstructed tensor.
        Union[None, torch.Tensor]
            The time mask tensor of shape [batch, num_channels, time].
        """
        if inference:
            latent, _, _ = self.forward_inference(x)
            return latent, None, None, None
        else:
            latent, mask, ids_restore, chan_embeds_encoder = self.forward_encoder(x, self.mask_percentage)
            pred, chan_embeds_decoder = self.forward_decoder(latent, mask, ids_restore)
            loss, reconstruction, time_mask = self.forward_loss(x, pred, mask)
            return latent, loss, reconstruction, time_mask
    
    @torch.no_grad()
    def _calculate_nuc_norm(self, embeddings: torch.Tensor) -> float:
        embeddings = embeddings.to(torch.float)
        _, S, _ = torch.linalg.svd(embeddings)
        nuc_norm = S.sum()
        nuc_norm = -1 * nuc_norm
        return nuc_norm
    
    @torch.no_grad()
    def _calculate_rankme(self, embeddings: torch.Tensor) -> float:
        embeddings = embeddings.to(torch.float)
        _, S, _ = torch.linalg.svd(embeddings)
        normalized_S = S / S.sum()
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-9))
        effective_rank = torch.exp(entropy)
        return effective_rank
    
    def _update_optimizer(self) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_scheduler.schedule[self.global_step]
        
    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int) -> torch.Tensor:
        # Update the optimizer
        self._update_optimizer()
        # Forward Pass
        _, loss, reconstruction, time_mask = self.forward(batch)
        # Plotting the reconstruction periodically
        if self.global_step % 100 == 0:
            raw_signal = batch[0]  #shape [1, num_cells, time]
            recon_signal = reconstruction[0].cpu().detach().numpy()  # shape [num_channels, time]
            raw_signal = raw_signal.cpu().detach().numpy() # shape: [num_cells, time]
            mask_plot = time_mask[0].cpu().detach().numpy()  # shape: [num_cells, time]
            expand_to = int(raw_signal.shape[1]*1.5)
            expand_to_per_cell = expand_to // self.num_channels
            raw_signal = np.repeat(raw_signal, expand_to_per_cell, axis=0)
            recon_signal = np.repeat(recon_signal, expand_to_per_cell, axis=0)
            mask_plot = np.repeat(mask_plot, expand_to_per_cell, axis=0)
            inv_mask_plot = ~mask_plot.astype(bool)
            inv_mask_plot = inv_mask_plot.astype(float)
            # Setting the masked values to the minimum value
            # So during plotting it shows up as black
            # This focuses only on the reconstructed values
            recon_signal[inv_mask_plot.astype(bool)] = np.amin(recon_signal)
            raw_signal_buff = self.create_collage(raw_signal, np.ones((raw_signal.shape[0], 100))*np.amin(raw_signal))
            recon_signal_buff = self.create_collage(recon_signal, np.ones((recon_signal.shape[0], 100))*np.amin(recon_signal))
            raw_signal_buff = (raw_signal_buff - np.amin(raw_signal_buff)) / (np.amax(raw_signal_buff) - np.amin(raw_signal_buff))
            recon_signal_buff = (recon_signal_buff - np.amin(recon_signal_buff)) / (np.amax(recon_signal_buff) - np.amin(recon_signal_buff))
            collage = self.create_collage(raw_signal_buff, recon_signal_buff)
            collage = self.create_collage(collage, inv_mask_plot)
            if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                self.logger.log_image(key="Reconstruction", images=[collage], step=self.global_step)
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step])
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Forward Pass
        _, loss, _, _ = self.forward(batch)
        self.log_dict({'val_loss': loss}, on_step=True, on_epoch=True)
            
    def configure_optimizers(self) -> torch.optim.Optimizer:
        param_groups = [{'params': self.parameters()}]
        optimizer = optim.AdamW(param_groups, lr=1e-3)
        # After warmup, use a scheduler of your choice
        return optimizer
    
    def on_train_start(self) -> None:
        # Computing the hyperparameter schedule
        assert self.trainer.max_epochs is not None, "The maximum number of epochs must be specified."
        train_steps = self.trainer.num_training_batches * self.trainer.max_epochs

        if dist.is_initialized():
            dataloader_lengths_tensor = [
                torch.tensor(0, device=self.device) for _ in range(self.trainer.world_size)
            ]
            dist.all_gather(
                dataloader_lengths_tensor, torch.tensor(len(self.trainer.train_dataloader), device=self.device)
            )
            dataloader_lengths = [dataloader_length.item() for dataloader_length in dataloader_lengths_tensor]
            assert len(set(dataloader_lengths)) == 1, "All dataloaders must have the same length"

        self.lr_scheduler.compute_schedule(self.trainer.train_dataloader)
        schedule_length = len(self.lr_scheduler.schedule)
        assert schedule_length != 0 and train_steps <= schedule_length
    
    def create_collage(self, image1: Union[torch.Tensor, np.ndarray], image2: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Create a collage of two images placed next to each other.
        
        Args:
            image1 (torch.Tensor or np.ndarray): The first image of shape [y, x].
            image2 (torch.Tensor or np.ndarray): The second image of shape [y, x].
            
        Returns:
            torch.Tensor or np.ndarray: The collage image of shape [y, x*2], with image1 and image2 side by side.
        """
        # Ensure both images have the same height
        assert image1.shape[0] == image2.shape[0], "Images must have the same height (y dimension)"
        
        # Concatenate the two images along the width (x-axis)
        collage_image = torch.cat((image1, image2), dim=1) if isinstance(image1, torch.Tensor) else np.concatenate((image1, image2), axis=1)
        
        return collage_image
    