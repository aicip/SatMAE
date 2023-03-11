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
        use_random_grayscale: bool = True,
        use_color_jitter: bool = True,
        color_jitter_intensity: float = 0.2,
        use_random_rotation: bool = True,
        rotation_degrees: float = 15,
        use_random_perspective: bool = True,
        perspective_distortion_scale: float = 0.15,
        use_gaussian_blur: bool = True,
        gaussian_blur_sigma: float = 5.0,
        gaussian_blur_kernel_size: int = 5,
        global_prob: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # applying random types of noise of a random intensity, as well as color jittering and other augmentations
        transforms = []
        # T.RandomGrayscale(p=0.1),
        #     T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.25),
        #     T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 5.0))], p=0.25),
        #     T.RandomApply([T.RandomRotation(degrees=15, expand=False)], p=0.25),
        #     T.RandomPerspective(distortion_scale=0.15, p=0.25),

        if use_random_grayscale:
            transforms.append(T.RandomGrayscale(p=global_prob))
        if use_color_jitter:
            transforms.append(
                T.RandomApply(
                    [
                        T.ColorJitter(
                            color_jitter_intensity,
                            color_jitter_intensity,
                            color_jitter_intensity,
                            color_jitter_intensity,
                        )
                    ],
                    p=global_prob,
                )
            )
        if use_random_rotation:
            transforms.append(
                T.RandomApply(
                    [T.RandomRotation(degrees=rotation_degrees, expand=False)],
                    p=global_prob,
                )
            )
        if use_random_perspective:
            transforms.append(
                T.RandomApply(
                    [
                        T.RandomPerspective(
                            distortion_scale=perspective_distortion_scale, p=global_prob
                        )
                    ]
                )
            )
        if use_gaussian_blur:
            transforms.append(
                T.RandomApply(
                    [
                        T.GaussianBlur(
                            gaussian_blur_kernel_size,
                            sigma=(gaussian_blur_sigma, gaussian_blur_sigma),
                        )
                    ],
                    p=global_prob,
                )
            )

        self.augment1 = nn.Sequential(*transforms)

    def forward(self, imgs, mask_ratio=0.75, mask_seed: int = None):
        if mask_seed is not None:
            torch.manual_seed(mask_seed)

        imgs = self.augment1(imgs)

        return super().forward(imgs, mask_ratio=mask_ratio, mask_seed=mask_seed)
