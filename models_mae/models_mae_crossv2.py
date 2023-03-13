import random

import torch
import torch.nn as nn
from torchvision import transforms as T

# xformers._is_functorch_available = True
import viz

from .models_mae import MaskedAutoencoderViT


class MaskedAutoencoderViTCrossV2(MaskedAutoencoderViT):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        loss_latent=None,
        loss_latent_weight: float = 1.0,
        losses_pred_reduction: str = "sum",
        lossed_latent_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_latent = loss_latent.lower() if loss_latent is not None else self.loss
        self.loss_latent_weight = loss_latent_weight
        assert (
            0.0 <= self.loss_latent_weight <= 1.0
        ), "loss_latent_weight must be in [0.0, 1.0]"

        self.losses_pred_reduction = losses_pred_reduction.lower()
        self.lossed_latent_reduction = lossed_latent_reduction.lower()

        allowed_reduction = ["mean", "sum"]
        assert (
            self.losses_pred_reduction in allowed_reduction
        ), f"losses_pred_reduction must be in {allowed_reduction}"
        assert (
            self.lossed_latent_reduction in allowed_reduction
        ), f"lossed_latent_reduction must be in {allowed_reduction}"

        print(f"Latent loss: {self.loss_latent} - weight: {self.loss_latent_weight}")
        print(f"Losses pred reduction: {self.losses_pred_reduction}")
        print(f"Lossed latent reduction: {self.lossed_latent_reduction}")

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

        losses_pred_reduced = sum(losses_pred)
        if self.losses_pred_reduction == "mean":
            losses_pred_reduced /= len(losses_pred)

        losses_latent_reduced = sum(losses_latent)
        if self.lossed_latent_reduction == "mean":
            losses_latent_reduced /= len(losses_latent)

        loss_final = (
            losses_pred_reduced + self.loss_latent_weight * losses_latent_reduced
        )

        return loss_final, pred_orig, mask_orig
