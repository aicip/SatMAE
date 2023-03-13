# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ShuntedTransformer: https://github.com/OliverRensu/Shunted-Transformer
# --------------------------------------------------------

from .models_mae import *
from .models_mae_aug import *
from .models_mae_cross import *
from .models_mae_shunted import *
from .models_mae_shunted_cross import *


# --- MAE Models --- #
def mae_vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug15(**kwargs):
    model = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        use_color_jitter=True,
        use_gaussian_blur=True,
        use_random_rotation=True,
        use_random_perspective=True,
        global_prob=0.15,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug25(**kwargs):
    model = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        use_color_jitter=True,
        use_gaussian_blur=True,
        use_random_rotation=True,
        use_random_perspective=True,
        global_prob=0.25,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug35(**kwargs):
    model = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        use_color_jitter=True,
        use_gaussian_blur=True,
        use_random_rotation=True,
        use_random_perspective=True,
        global_prob=0.35,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug25_nrp(**kwargs):
    model: MaskedAutoencoderViTAug = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        global_prob=0.25,
        use_color_jitter=True,
        use_gaussian_blur=True,
        use_random_rotation=False,
        use_random_perspective=False,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug25_rp(**kwargs):
    model: MaskedAutoencoderViTAug = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        global_prob=0.25,
        use_color_jitter=False,
        use_gaussian_blur=False,
        use_random_rotation=True,
        use_random_perspective=True,
        **kwargs,
    )
    return model


def mae_vit_tiny_aug50_rrc(**kwargs):
    model = MaskedAutoencoderViTAug(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Augmentation
        use_random_resized_crop=True,
        random_resized_crop_scale=(0.2, 0.8),
        use_color_jitter=False,
        use_gaussian_blur=False,
        use_random_rotation=False,
        use_random_perspective=False,
        global_prob=0.50,
        **kwargs,
    )
    return model


def mae_vit_tiny_cross(**kwargs):
    model = MaskedAutoencoderViTCross(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        predictor_hidden_size=1024,
        **kwargs,
    )
    return model


def mae_vit_mini(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=256,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def mae_vit_small(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=512,
        encoder_num_layers=8,
        encoder_num_heads=8,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def mae_vit_small_cross(**kwargs):
    model = MaskedAutoencoderViTCross(
        dim_model=512,
        encoder_num_layers=8,
        encoder_num_heads=8,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        predictor_hidden_size=2048,
        **kwargs,
    )
    return model


def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=768,
        encoder_num_layers=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def mae_vit_base_cross(**kwargs):
    model = MaskedAutoencoderViTCross(
        dim_model=768,
        encoder_num_layers=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        predictor_hidden_size=2048,
        **kwargs,
    )
    return model


def mae_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=1024,
        encoder_num_layers=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def mae_vit_huge(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=1280,
        encoder_num_layers=32,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


# --- Shunted Models --- #
def mae_vit_tiny_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[64, 128],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_tiny = mae_vit_tiny_shunted_2st


def mae_vit_tiny_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[64, 128],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_tiny_cross = mae_vit_tiny_shunted_2st_cross


def mae_vit_mini_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[128, 256],
        encoder_num_layers=[2, 4],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[4, 2],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_mini = mae_vit_mini_shunted_2st


def mae_vit_small_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[256, 512],
        encoder_num_layers=[4, 8],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_small = mae_vit_small_shunted_2st


def mae_vit_small_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[256, 512],
        encoder_num_layers=[4, 8],
        encoder_num_heads=[4, 8],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_small_cross = mae_vit_small_shunted_2st_cross


def mae_vit_base_shunted_2st(**kwargs):
    model = MaskedAutoencoderShuntedViT(
        # Encoder
        dim_model=[512, 768],
        encoder_num_layers=[8, 12],
        encoder_num_heads=[8, 12],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_base = mae_vit_base_shunted_2st


def mae_vit_base_shunted_2st_cross(**kwargs):
    model = MaskedAutoencoderShuntedViTCross(
        # Encoder
        dim_model=[512, 768],
        encoder_num_layers=[8, 12],
        encoder_num_heads=[8, 12],
        mlp_ratios=[4, 4],
        sr_ratios=[2, 1],
        # Decoder
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


shunted_2s_mae_vit_base_cross = mae_vit_base_shunted_2st_cross
