# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ShuntedTransformer: https://github.com/OliverRensu/Shunted-Transformer
# --------------------------------------------------------

from .models_mae import *
from .models_mae_cross import *
from .models_mae_crossv2 import *
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


def mae_vit_tiny_crossV2_psum_lsum_full(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="sum",
        lossed_latent_reduction="sum",
        loss_latent_weight=1.0,
        **kwargs,
    )
    return model


def mae_vit_tiny_crossV2_psum_lsum_half(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="sum",
        lossed_latent_reduction="sum",
        loss_latent_weight=0.5,
        **kwargs,
    )
    return model


def mae_vit_tiny_crossV2_pmean_lmean_full(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="mean",
        lossed_latent_reduction="mean",
        loss_latent_weight=1.0,
        **kwargs,
    )
    return model


def mae_vit_tiny_crossV2_pmean_lmean_half(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="mean",
        lossed_latent_reduction="mean",
        loss_latent_weight=0.5,
        **kwargs,
    )
    return model

def mae_vit_tiny_crossV2_psum_lmean_full(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="sum",
        lossed_latent_reduction="mean",
        loss_latent_weight=1.0,
        **kwargs,
    )
    return model

def mae_vit_tiny_crossV2_psum_lmean_half(**kwargs):
    model = MaskedAutoencoderViTCrossV2(
        dim_model=128,
        encoder_num_layers=4,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_num_layers=4,
        decoder_num_heads=8,
        # Cross Args
        losses_pred_reduction="sum",
        lossed_latent_reduction="mean",
        loss_latent_weight=0.5,
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
