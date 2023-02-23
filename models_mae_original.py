# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()
        print(f"img_size: {img_size}")
        print(f"patch_size: {patch_size}")
        print(f"in_chans: {in_chans}")
        print(f"embed_dims: {embed_dim}")
        print(f"depths: {depth}")
        print(f"num_heads: {num_heads}")
        print(f"mlp_ratios: {mlp_ratio}")
        print(f"decoder_embed_dim: {decoder_embed_dim}")
        print(f"decoder_depth: {decoder_depth}")
        print(f"decoder_num_heads: {decoder_num_heads}")
        print("--"*8, "Init", "--"*8)

        self.in_c = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        print(f"num_patches: {num_patches}")
        print(f"patche embed img_size: {self.patch_embed.img_size}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 
                        num_patches + 1, 
                        decoder_embed_dim), 
            requires_grad=False
        )  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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
        print("**** Patchify +++")
        # p = self.patch_embed.patch_size[0]
        print(f"imgs.shape: {imgs.shape}, p: {p}, c: {c}")
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
        print("**** Random masking ****")
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        print(f"ids_shuffle.shape: {ids_shuffle.shape}")
        print(f"ids_restore.shape: {ids_restore.shape}")

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        print(f"x_masked.shape: {x_masked.shape}")

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print(f"mask.shape: {mask.shape}")
        print("***********************")
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        print("---"*8, "forward_encoder", "---"*8)
        print(f"Original x.shape: {x.shape}")
        # embed patches
        print(f"Expected patch_embed.img_size: {self.patch_embed.img_size}")
        x = self.patch_embed(x)
        print(f"patch_embed.x.shape: {x.shape}")

        # add pos embed w/o cls token
        print(f"Expected pos_embed.shape: {self.pos_embed[:, 1:, :].shape}")
        x = x + self.pos_embed[:, 1:, :]
        print(f"pos_embed.x.shape: {x.shape}")

        # masking: length -> length * mask_ratio
        print(f"mask_ratio: {mask_ratio}")
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        print(f"random_masking.x.shape: {x.shape}")
        print(f"ids_restore.shape: {ids_restore.shape}, mask.shape: {mask.shape}")
        # append cls token
        print(f"Expected cls_token.shape: {self.cls_token.shape}")
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        print(f"cls_token+pos_embed.shape: {cls_token.shape}")
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        print(f"cls_token_expand.shape: {cls_tokens.shape}")
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"cls_token.x.shape: {x.shape}")

        # apply Transformer blocks
        print("###", "Transformer blocks", "###")
        for blk_ind, blk in enumerate(self.blocks):
            # print(f"\tInputs to block_{blk_ind}: (x.shape): {x.shape})")
            # print(f"\t\tPatch Embed img_size: {self.patch_embed.img_size}")
            x = blk(x)
            print(f"\tOutputs from block_{blk_ind}: (x.shape): {x.shape})")
        x = self.norm(x)
        print(f"norm.x.shape: {x.shape}")
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        print("---"*8, " Decoder ", "---"*8)
        print(f"Original x.shape: {x.shape}")
        print(f"ids_restore.shape: {ids_restore.shape}")
        # embed tokens
        x = self.decoder_embed(x)
        print(f"decoder_embed.x.shape: {x.shape}")

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        print(f"mask_tokens.shape: {mask_tokens.shape}")
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        print(f"x_.shape: {x_.shape}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        print(f"x_.shape: {x_.shape}")
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        print(f"x.shape: {x.shape}")

        # add pos embed
        print(f"x + decoder_pos_embed: {x.shape} + {self.decoder_pos_embed.shape}")
        x = x + self.decoder_pos_embed
        print(f"decoder_pos_embed+x.shape: {x.shape}")

        # apply Transformer blocks
        for blk_ind, blk in enumerate(self.decoder_blocks):
            print(f"Block_{blk_ind}: (x.shape): -> In {x.shape})")
            print(f"    Patch Embed img_size: {self.patch_embed.img_size}")
            x = blk(x)
            print(f"    Outputs from block_{blk_ind}: (x.shape): {x.shape})")
        x = self.decoder_norm(x)
        print(f"norm.x.shape: {x.shape}")

        # predictor projection
        x = self.decoder_pred(x)
        print(f"decoder_pred.x.shape: {x.shape}")

        # remove cls token
        x = x[:, 1:, :]
        print(f"out.x.shape: {x.shape}")

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        print("###"*8, " Loss ", "###"*8)
        print(f"mask.shape: {mask.shape}")
        print(f"pred.shape: {pred.shape}")
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        print(f"imgs.shape: {imgs.shape}")
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        print(f"patchify.target.shape: {target.shape}")
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
        raise NotImplementedError("Should test different configs first.")
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        # embed_dim=1024,
        # depth=24,
        # num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        # mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
shunted_mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
