# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from xformers.factory import xFormer, xFormerConfig


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        dim_model=1024,
        # Encoder parameters
        encoder_num_layers=24,
        encoder_num_heads=16,
        # Decoder parameters
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        # Residual parameters
        residual_norm_style="post",
        residual_dropout=0.0,
        # Feedforward parameters
        feedforward="MLP",
        feedforward_activation="gelu",
        feedforward_hidden_layer_multiplier=4.0,
        feedforward_dropout=0.0,
        # Attention parameters
        attention=None, # Passed from pretrain script
        attention_dropout=0.0,
        # Other parameters
        reversible=False,
        norm_pix_loss=False,
        norm_layer=nn.LayerNorm,  # TODO: This is not used anymore (check if XFormer has it for sure)
    ):
        super().__init__()

        self.in_c = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        assert img_size % patch_size == 0

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim_model)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim_model), requires_grad=False
        )  # fixed sin-cos embedding

        num_patches = (img_size // patch_size) ** 2

        encoder_xformer_config = [
            {
                "reversible": reversible,  # This decreases memory usage but increases latency
                "block_type": "encoder",
                "num_layers": encoder_num_layers,
                "dim_model": dim_model,
                "residual_norm_style": residual_norm_style,
                "multi_head_config": {
                    "num_heads": encoder_num_heads,
                    "residual_dropout": residual_dropout,
                    "attention": {
                        "name": attention,
                        "dropout": attention_dropout,
                        "seq_len": num_patches + 1,  # This adds the mask token
                        "causal": False,  # TODO: Check if needs to be True
                        # "use_rotary_embeddings": True, # TODO: Check if this would be useful
                    },
                },
                "feedforward_config": {
                    "name": feedforward,
                    "dropout": feedforward_dropout,
                    "activation": feedforward_activation,
                    "hidden_layer_multiplier": feedforward_hidden_layer_multiplier,
                },
            }
        ]

        encoder_config = xFormerConfig(encoder_xformer_config)
        self.encoder = xFormer.from_config(encoder_config)
        # TODO: The norm may already be handled by XFormer (check this)
        # self.encoder_norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dim_model, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_xformer_config = [
            {
                "reversible": reversible,
                # Using encoder here since the rest of the decoder parts are handled manually (see below)
                "block_type": "encoder",
                "num_layers": decoder_num_layers,
                "dim_model": decoder_embed_dim,
                "residual_norm_style": residual_norm_style,
                "multi_head_config": {
                    "num_heads": decoder_num_heads,
                    "residual_dropout": residual_dropout,
                    "attention": {
                        "name": attention,
                        "dropout": attention_dropout,
                        "seq_len": num_patches + 1,  # This adds the mask token
                        "causal": False,
                        # "use_rotary_embeddings": True, # TODO: Check if this would be useful
                    },
                },
                "feedforward_config": {
                    "name": feedforward,
                    "dropout": feedforward_dropout,
                    "activation": feedforward_activation,
                    "hidden_layer_multiplier": feedforward_hidden_layer_multiplier,
                },
            }
        ]

        decoder_config = xFormerConfig(decoder_xformer_config)
        self.decoder = xFormer.from_config(decoder_config)
        # TODO: The norm may already be handled by XFormer (check this)
        # self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(encoder_pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.encoder_pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # for blk in self.encoder_blocks:
        #     x = blk(x)
        x = self.encoder(x)
        # x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        x = self.decoder(x)
        # x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=768,
        encoder_num_layers=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        feedforward_hidden_layer_multiplier=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=1024,
        encoder_num_layers=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        feedforward_hidden_layer_multiplier=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        dim_model=1280,
        encoder_num_layers=32,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
        feedforward_hidden_layer_multiplier=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
