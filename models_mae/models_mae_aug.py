import random

import torch
import torch.nn as nn
from torchvision import transforms as T

# xformers._is_functorch_available = True
import viz

from .models_mae import MaskedAutoencoderViT


class MaskedAutoencoderViTAug(MaskedAutoencoderViT):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_latent=None,
        loss_latent_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_latent = loss_latent.lower() if loss_latent is not None else self.loss
        self.loss_latent_weight = loss_latent_weight
        print(f"Latent loss: {self.loss_latent} - weight: {self.loss_latent_weight}")

        self.crop_sm = nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.1, 0.4),
                antialias=True,
            )
        )

        self.crop_md = nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.4, 0.7),
                antialias=True,
            )
        )

        # T.RandomApply(
        #     [
        #         T.RandomResizedCrop(
        #             size=(self.input_size, self.input_size),
        #             scale=random_resized_crop_scale,
        #             antialias=True,
        #         )
        #     ],
        #     p=global_prob,
        # )

        # T.RandomApply(
        #     [
        #         T.ColorJitter(
        #             color_jitter_intensity,
        #             color_jitter_intensity,
        #             color_jitter_intensity,
        #             color_jitter_intensity,
        #         )
        #     ],
        #     p=global_prob,
        # )

        # T.RandomApply(
        #     [T.RandomRotation(degrees=rotation_degrees, expand=False)],
        #     p=global_prob,
        # )

        # T.RandomApply(
        #     [
        #         T.RandomPerspective(
        #             distortion_scale=perspective_distortion_scale, p=global_prob
        #         )
        #     ]
        # )

        # T.RandomApply(
        #     [
        #         T.GaussianBlur(
        #             gaussian_blur_kernel_size,
        #             sigma=(gaussian_blur_sigma, gaussian_blur_sigma),
        #         )
        #     ],
        #     p=global_prob,
        # )

        # self.augment1 = nn.Sequential(*transforms)

    def forward(self, imgs, mask_ratio=0.75, mask_seed: int = None):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)

        imgs_crop_sm, imgs_crop_md = self.crop_sm(imgs), self.crop_md(imgs)

        loss_crop_sm, _, _, latent_crop_sm = super().forward(
            imgs_crop_sm, mask_ratio=mask_ratio, mask_seed=mask_seed, return_latent=True
        )
        loss_crop_md, _, _, latent_crop_md = super().forward(
            imgs_crop_md, mask_ratio=mask_ratio, mask_seed=mask_seed, return_latent=True
        )
        loss_orig, pred_orig, mask_orig, latent_orig = super().forward(
            imgs, mask_ratio=mask_ratio, mask_seed=mask_seed, return_latent=True
        )

        losses_pred = [loss_orig, loss_crop_sm, loss_crop_md]

        # latent loss between each crop and original
        # as well as between crops themselves
        latent_loss_fn = None
        if self.loss_latent == "mse":
            latent_loss_fn = nn.MSELoss(reduction="mean")
        elif self.loss_latent == "l1":
            latent_loss_fn = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unknown latent loss: {self.latent_loss}")

        losses_latent = [
            latent_loss_fn(latent_crop_sm, latent_orig),
            latent_loss_fn(latent_crop_md, latent_orig),
            latent_loss_fn(latent_crop_sm, latent_crop_md),
        ]

        print(f"losses_pred: {losses_pred}")
        print(f"losses_latent: {losses_latent}")

        loss_final = sum(losses_pred) + self.loss_latent_weight * sum(losses_latent)

        return loss_final, pred_orig, mask_orig
