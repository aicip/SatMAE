from typing import Union, Tuple
import random
import torch
import torch.nn as nn
from torchvision import transforms as T

from .models_mae_shunted import MaskedAutoencoderShuntedViT
from .models_mae_cross import MLP, RandomApply, default


class MaskedAutoencoderShuntedViTCross(MaskedAutoencoderShuntedViT):
    """Cross-Prediction Masked Autoencoder
    with Shunted VisionTransformer backbone"""

    def __init__(
        self,
        augment_fn1: Union[nn.Sequential, None] = None,
        augment_fn2: Union[nn.Sequential, None] = None,
        predictor_hidden_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # augmentation define
        # Adding two augments for the input to get subsampled image x1, x2: x1 = x, x2 = subsample(x)
        AUG1 = torch.nn.Sequential(T.Resize(size=(self.input_size, self.input_size)))
        AUG2 = torch.nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.2, 0.8),
                antialias=True,
            )
        )

        self.augment1 = default(augment_fn1, AUG1)
        self.augment2 = default(augment_fn2, AUG2)
        # may need to be modified to keep size consistency
        patch_embed = getattr(self, f"patch_embed1")
        # patch_embed = getattr(self, f"patch_embed{self.num_stages}")
        self.predictor = MLP(
            self.decoder_embed_dim,
            patch_embed.num_patches,
            predictor_hidden_size,
        )

    def forward(
        self,
        imgs: torch.Tensor,
        mask_seed: Union[None, int] = None,
        consistent_mask: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1, img2 = self.augment1(imgs), self.augment2(imgs)

        if mask_seed is None and consistent_mask:
            mask_seed = int(torch.randint(0, 100000000, (1,)).item())

        if mask_seed is not None:
            torch.manual_seed(mask_seed)

        latent1, mask1, ids_restore1 = self.forward_encoder(img1)

        if mask_seed is not None and consistent_mask:
            torch.manual_seed(mask_seed)

        latent2, mask2, ids_restore2 = self.forward_encoder(img2)

        pred1, dec_emd_1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
        pred2, dec_emd_2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]
    
        
        if self.loss == "mse":
            loss1 = self.forward_loss_mse(img1, pred1, mask1)
            loss2 = self.forward_loss_mse(img2, pred2, mask2)
        elif self.loss == "l1":
            loss1 = self.forward_loss_l1(img1, pred1, mask1)
            loss2 = self.forward_loss_l1(img2, pred2, mask2)
        else:
            raise ValueError(f"Loss type {self.loss} not supported.")

        c_loss = nn.MSELoss()
        cross_pred = self.predictor(dec_emd_2[:, 1:, :])
        cross_loss = c_loss(dec_emd_1[:, 1:, :], cross_pred)

        loss = loss1 + loss2 + cross_loss

        if self.print_level > 1:
            raise Exception("Stopping because you set the print_level > 1.")
        return loss, pred1, mask1
