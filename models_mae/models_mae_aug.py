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
        # Random resized crop
        use_random_resized_crop: bool = False,
        random_resized_crop_scale: tuple = (0.2, 0.8),
        # Color-related (brightness, contrast, saturation, hue, grayscale)
        use_random_grayscale: bool = False,
        use_color_jitter: bool = False,
        color_jitter_intensity: float = 0.2,
        # Gaussian blur
        use_gaussian_blur: bool = False,
        gaussian_blur_sigma: float = 5.0,
        gaussian_blur_kernel_size: int = 5,
        # Rotation and perspective
        use_random_rotation: bool = False,
        rotation_degrees: float = 15,
        use_random_perspective: bool = False,
        perspective_distortion_scale: float = 0.15,
        # Probability of applying each augmentation
        global_prob: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)

        transforms = []

        if use_random_resized_crop:
            transforms.append(
                T.RandomApply(
                    [
                        T.RandomResizedCrop(
                            size=(self.input_size, self.input_size),
                            scale=random_resized_crop_scale,
                            antialias=True,
                        )
                    ],
                    p=global_prob,
                )
            )

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
