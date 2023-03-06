# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import xformers
from timm.models.vision_transformer import Block, PatchEmbed
from xformers.factory import xFormer, xFormerConfig

from util.pos_embed import get_2d_sincos_pos_embed

# xformers._is_functorch_available = True

# adding two function, MLP is for prediction, RandomApply is for augment

def MLP(dim, projection_size=512, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def default(val, def_val):
    return def_val if val is None else val


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        input_size=128,
        input_channels=3,
        patch_size=16,  # Must be multiple of input_size
        dim_model=1024,
        # Encoder parameters
        encoder_num_layers=24,
        encoder_num_heads=16,  # Must be multiple of dim_model
        # Decoder parameters
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,  # Must be multiple of decoder_embed_dim
        # Residual parameters
        residual_norm_style="post",
        residual_dropout=0.0,
        # Feedforward parameters
        ffn_name="MLP",  # Note: If use_xformers=False, only MLP is supported
        ffn_activation="gelu",  # Note: if use_xformers=False, only gelu is supported
        ffn_ratio=4,
        ffn_dropout=0.0,
        # Attention parameters
        attn_name="scaled_dot_product",
        attn_dropout=0.0,
        # Other parameters
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6
        ),  # Note: Only used if use_xformers=False
        norm_pix_loss=False,
        use_xformers=False,
        loss="mse",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.loss = loss.lower()

        self.use_xformers = use_xformers

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        assert input_size % patch_size == 0

        if not use_xformers:
            assert (
                attn_name == "scaled_dot_product"
            ), f"Attention {attn_name} not supported with use_xformers=False, as Timm's implementation uses scaled_dot_product"
            assert (
                ffn_name == "MLP"
            ), f"Feedforward {ffn_name} not supported with use_xformers=False, as Timm's implementation uses MLP"
            assert (
                ffn_activation == "gelu"
            ), f"Feedforward activation {ffn_activation} not supported with use_xformers=False, as Timm's implementation uses gelu"

        # augmentation define
        # Adding two augments for the input to get subsampled image x1, x2: x1 = x, x2 = subsample(x)
        AUG1 = torch.nn.Sequential(
            T.Resize(size=(input_size, input_size))
        )
        AUG2 = torch.nn.Sequential(
            T.RandomResizedCrop(size=(input_size, input_size), scale=(0.2, 0.8))
            # T.Resize(size=img_size)
        )

        self.augment1 = default(augment_fn1, AUG1)
        self.augment2 = default(augment_fn2, AUG2)
        
        self.patch_embed = PatchEmbed(input_size, patch_size, input_channels, dim_model)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim_model), requires_grad=False
        )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dim_model, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        if use_xformers:
            print("Using xformers")
            encoder_config = xFormerConfig(
                [
                    {
                        "reversible": False,  # This decreases memory usage but increases latency
                        "block_type": "encoder",
                        "num_layers": encoder_num_layers,
                        "dim_model": dim_model,
                        "residual_norm_style": residual_norm_style,
                        "multi_head_config": {
                            "num_heads": encoder_num_heads,
                            "residual_dropout": residual_dropout,
                            "attention": {
                                "name": attn_name,
                                "dropout": attn_dropout,
                                "seq_len": num_patches + 1,  # This adds the mask token
                                "causal": False,
                                "use_rotary_embeddings": False,  # TODO: Check if this would be useful
                            },
                        },
                        "feedforward_config": {
                            "name": ffn_name,
                            "dropout": ffn_dropout,
                            "activation": ffn_activation,
                            "hidden_layer_multiplier": ffn_ratio,
                        },
                    }
                ]
            )
            self.encoder = xFormer.from_config(encoder_config)

            decoder_config = xFormerConfig(
                [
                    {
                        "reversible": False,
                        # Using encoder here since the rest of the decoder parts are handled manually (see below)
                        "block_type": "encoder",
                        "num_layers": decoder_num_layers,
                        "dim_model": decoder_embed_dim,
                        "residual_norm_style": residual_norm_style,
                        "multi_head_config": {
                            "num_heads": decoder_num_heads,
                            "residual_dropout": residual_dropout,
                            "attention": {
                                "name": attn_name,
                                "dropout": attn_dropout,
                                "seq_len": num_patches + 1,  # This adds the mask token
                                "causal": False,
                                "use_rotary_embeddings": False,  # TODO: Check if this would be useful
                            },
                        },
                        "feedforward_config": {
                            "name": ffn_name,
                            "dropout": ffn_dropout,
                            "activation": ffn_activation,
                            "hidden_layer_multiplier": ffn_ratio,
                        },
                    }
                ]
            )
            self.decoder = xFormer.from_config(decoder_config)
        else:
            print("Using Timm")
            encoder_blocks = [
                Block(
                    dim=dim_model,
                    num_heads=encoder_num_heads,
                    mlp_ratio=ffn_ratio,
                    qkv_bias=True,
                    drop=ffn_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=norm_layer,
                    drop_path=residual_dropout,
                )
                for _ in range(encoder_num_layers)
            ]
            encoder_norm = nn.LayerNorm(dim_model)
            if residual_norm_style == "post":
                encoder_blocks.append(encoder_norm)
            elif residual_norm_style == "pre":
                encoder_blocks.insert(0, encoder_norm)
            else:
                raise ValueError(
                    f"residual_norm_style: {residual_norm_style} not supported"
                )

            self.encoder = nn.ModuleList(encoder_blocks)

            decoder_blocks = [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=ffn_ratio,
                    qkv_bias=True,
                    drop=ffn_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=norm_layer,
                    drop_path=residual_dropout,
                )
                for _ in range(decoder_num_layers)
            ]
            decoder_norm = nn.LayerNorm(decoder_embed_dim)
            if residual_norm_style == "post":
                decoder_blocks.append(decoder_norm)
            elif residual_norm_style == "pre":
                decoder_blocks.insert(0, decoder_norm)
            else:
                raise ValueError(
                    f"residual_norm_style: {residual_norm_style} not supported"
                )

            self.decoder = nn.ModuleList(decoder_blocks)

        # decoder to patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * input_channels, bias=True
        )
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
        # TODO: Test out adding random noise to the input
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
        if self.use_xformers:
            x = self.encoder(x)
        else:
            for blk in self.encoder:
                x = blk(x)

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
        if self.use_xformers:
            x = self.decoder(x)
        else:
            for blk in self.decoder:
                x = blk(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss_mse(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        target = self.patchify(
            imgs, self.patch_embed.patch_size[0], self.input_channels
        )
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # print("pred", pred.shape)
        # torch.Size([512, 64, 192])

        # print("target", target.shape)
        # torch.Size([512, 64, 192])

        loss = (pred - target) ** 2
        # print("loss", loss.shape)
        # torch.Size([512, 64, 192])

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print("loss", loss.shape)
        # torch.Size([512, 64])

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward_loss_l1(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(
            imgs, self.patch_embed.patch_size[0], self.input_channels
        )
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # print("pred", pred.shape)
        # torch.Size([512, 64, 192])

        # print("target", target.shape)
        # torch.Size([512, 64, 192])

        # final shape should be [512, 64] before (loss * mask).sum() / mask.sum()
        loss = torch.abs(pred - target)
        # print("loss", loss.shape)
        # torch.Size([512, 64, 192])

        # mean loss per patch
        loss = loss.mean(dim=-1)
        # print("loss", loss.shape)
        # torch.Size([512, 64])

        # mean loss on removed patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # TODO: Add flag for loss function
        if self.loss == "mse":
            loss = self.forward_loss_mse(imgs, pred, mask)
        elif self.loss == "l1":
            loss = self.forward_loss_l1(imgs, pred, mask)
        else:
            raise ValueError(f"Loss type {self.loss} not supported.")
        # loss = self.forward_loss_l1(imgs, pred, mask)
        return loss, pred, mask


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
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_num_layers=8,
        decoder_num_heads=16,
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
