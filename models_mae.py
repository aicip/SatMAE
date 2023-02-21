# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from torch import _assert
import collections.abc
from itertools import repeat
from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from shunted import Block as ShuntedBlock, Head as ShuntedHead, OverlapPatchEmbed


# TODO: rename to MaskedShuntedAutoencoderViT
class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 # embed_dim=1024, # replaced
                 # depth=24, # replaced
                 # num_heads=16, # replaced
                 decoder_embed_dim=512,
                 decoder_depth=8,  # replaced
                 decoder_num_heads=16,
                 # mlp_ratio=4.0, # replaced
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 # shunted arguments
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 num_conv=0
                 ):
        super().__init__()
        print(f"img_size: {img_size}")
        print(f"patch_size: {patch_size}")
        print(f"in_chans: {in_chans}")
        print(f"embed_dims: {embed_dims}")
        print(f"num_heads: {num_heads}")
        print(f"mlp_ratios: {mlp_ratios}")
        print(f"decoder_embed_dim: {decoder_embed_dim}")
        print(f"decoder_depth: {decoder_depth}")
        print(f"decoder_num_heads: {decoder_num_heads}")
        print(f"depths: {depths}")
        print(f"sr_ratios: {sr_ratios}")
        print(f"num_conv: {num_conv}")
        print("---"*20)

        self.in_c = in_chans
        self.depths = depths  # shunted
        self.num_stages = num_stages  # shunted

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        # --- START replaced code --- #
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding
        # self.blocks = nn.ModuleList(
        #     [
        #         Block(
        #             embed_dim,
        #             num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             norm_layer=norm_layer,
        #         )
        #         for _ in range(depth)
        #     ]
        # )
        # self.norm = norm_layer(embed_dim)
        # --- END replaced code --- #

        # --- START shunted equiv code --- #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        num_patches = []
        for i in range(num_stages):
            if i == 0 and False:  # TODO: temporarily disabled
                # TODO: Can't get number of patches from this
                patch_embed = ShuntedHead(num_conv)
            else:  # TODO: should change the sizes for every stage but keep it as it is until pipeline is working
                # patch_embed = PatchEmbed(img_size=img_size,
                #                          patch_size=patch_size,
                #                          in_chans=in_chans,
                #                          embed_dim=embed_dims[i])
                patch_embed = DefaultPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 2)),
                                                patch_size=patch_size,# if i == 0 else 3,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i])
                # patch_embed = OverlapPatchEmbed(img_size=img_size,
                #                                 patch_size=patch_size,
                #                                 stride=patch_size,
                #                                 in_chans=in_chans,
                #                                 embed_dim=embed_dims[i])
                # patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                #                                 # patch_size=7 if i == 0 else 3, # TODO: this is from shunted, but not sure if it's correct
                #                                 patch_size=patch_size if i == 0 else 3, # TODO: is this correct?
                #                                 # stride=4 if i == 0 else 2,
                #                                 stride=patch_size if i == 0 else 3,
                #                                 in_chans=in_chans if i == 0 else embed_dims[i - 1],
                #                                 embed_dim=embed_dims[i])

            blocks = nn.ModuleList([ShuntedBlock(dim=embed_dims[i],
                                                 num_heads=num_heads[i],
                                                 mlp_ratio=mlp_ratios[i],
                                                 qkv_bias=True,
                                                 drop=drop_rate,
                                                 attn_drop=attn_drop_rate,
                                                 drop_path=dpr[cur + j],
                                                 norm_layer=norm_layer,
                                                 sr_ratio=sr_ratios[i])
                                    for j in range(depths[i])])
            # blocks = nn.ModuleList([Block(dim=embed_dims[i],
            #                              num_heads=num_heads[i],
            #                              mlp_ratio=mlp_ratios[i],
            #                              qkv_bias=True,
            #                              norm_layer=norm_layer)
            #                         for _ in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            num_patches.append(patch_embed.num_patches)
            # cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[i]))
            pos_embed = nn.Parameter(torch.zeros(1,
                                                 num_patches[i] + 1,
                                                 embed_dims[i]),
                                     requires_grad=False)  # fixed sin-cos embedding

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"blocks{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
            # setattr(self, f"cls_token{i + 1}", cls_token)  # TODO: Shunted is not using this
            # TODO: Shunted is not using this
            setattr(self, f"pos_embed{i + 1}", pos_embed)
        # Note: Replace the orignal patch_embed, block, norm, cls_token, pos_embed (self vars)
        # With the self vars:
        # patch_embed1, patch_embed2, patch_embed3, patch_embed4
        # block1, block2, block3, block4
        # norm1, norm2, norm3, norm4
        # cls_token1, cls_token2, cls_token3, cls_token4
        # pos_embed1, pos_embed2, pos_embed3, pos_embed4
        # --- END shunted equiv code --- #

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dims[-1],  # replaced with shunted equiv
                                       decoder_embed_dim,
                                       bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1,
                                                          num_patches[0] + 1,
                                                          decoder_embed_dim),
                                              requires_grad=False
                                              )  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    # replaced with shunted equiv (mlp_ratios default: all 4)
                    mlp_ratios[-1],
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
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

        # --- START replaced code --- #
        # # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.pos_embed.shape[-1],
        #     int(self.patch_embed.num_patches**0.5),
        #     cls_token=True,
        # )
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1],
        #     int(self.patch_embed.num_patches**0.5),
        #     cls_token=True,
        # )
        # self.decoder_pos_embed.data.copy_(
        #     torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        # )
        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        # --- END replaced code --- #

        # --- START shunted equiv code --- #
        # for i in range(self.num_stages):
        #     self_patch_embed = getattr(self, f"patch_embed{i + 1}")
        #     self_pos_embed = getattr(self, f"pos_embed{i + 1}")
        #     # initialize (and freeze) pos_embed by sin-cos embedding
        #     pos_embed = get_2d_sincos_pos_embed(
        #         self_pos_embed.shape[-1],
        #         int(self_patch_embed.num_patches**0.5),
        #         cls_token=True,
        #     )
        #     setattr(getattr(self, f"pos_embed{i + 1}"), "data", torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch_ind = self.num_stages-1
        patch_ind = 0
        self_patch_embed = getattr(self, f"patch_embed{patch_ind+1}")
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            # replaced with shunted equiv
            int(self_patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        for i in range(self.num_stages):  # TODO: move up? order matters?
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            w = self_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            # torch.nn.init.normal_(getattr(self, f"cls_token{i + 1}"), std=0.02)
            torch.nn.init.normal_(self.mask_token, std=0.02)
        # --- END shunted equiv code --- #

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
        elif isinstance(m, nn.Conv2d):  # from shunted code
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

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
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        print(f"x_masked.shape: {x_masked.shape}")
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print(f"mask.shape: {mask.shape}")

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # --- START replaced code --- #
        # # embed patches
        # x, H, W  = self.patch_embed1(x)
        # print(f"patch_embed1.x.shape: {x.shape}")

        # # add pos embed w/o cls token
        # print(f"pos_embed1.shape: {self.pos_embed1[:, 1:, :].shape}")
        # x = x + self.pos_embed1[:, 1:, :]
        # print(f"pos_embed1.x.shape: {x.shape}")

        # # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print(f"random_masking1.x.shape: {x.shape}")

        # # append cls token
        # cls_token = self.cls_token1 + self.pos_embed1[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(f"cls_token1.x.shape: {x.shape}")

        # # apply Transformer blocks
        # for blk_ind, blk in enumerate(self.blocks1):
        #     x = blk(x)
        #     print(f"block1_{blk_ind}.x.shape: {x.shape}")
        # x = self.norm(x)
        # raise NotImplementedError
        # --- END replaced code --- #

        # --- START shunted equiv code --- #
        B = x.shape[0]
        print("---"*20)
        print(f"Original x.shape: {x.shape}")
        for i in range(self.num_stages):
            print("++"*10)
            print(f"Stage {i+1}:")
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            self_pos_embed = getattr(self, f"pos_embed{i + 1}")
            # self_cls_token = getattr(self, f"cls_token{i + 1}")
            self_blocks = getattr(self, f"blocks{i + 1}")
            self_norm = getattr(self, f"norm{i + 1}")

            # embed patches
            # x  = self_patch_embed(x)
            print(f"Expected patch_embed{i+1}.img_size: {self_patch_embed.img_size}")
            x, H, W = self_patch_embed(x)
            print(f"patch_embed{i + 1}.x.shape: {x.shape}")
            print(f"B, H, W: {B, H, W}")

            # add pos embed w/o cls token
            print(f"pos_embed{i + 1}.shape: {self_pos_embed[:, 1:, :].shape}")
            x = x + self_pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            # TODO: should I keep all masks and ids or maybe I should only create one mask?
            if i==0:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
                print(f"random_masking{i + 1}.x.shape: {x.shape}")
                H, W = int(H/2), int(W/2)
                print(f"B, H, W: {B, H, W}")
            # _, _, H, W = self_patch_embed.proj(x).shape
            # print(f"[Custom projection] B, H, W: {B, H, W}")
            # append cls token
            # cls_token = self_cls_token + self_pos_embed[:, :1, :]
            # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            # x = torch.cat((cls_tokens, x), dim=1)
            # print(f"cls_token{i + 1}.x.shape: {x.shape}")

            # apply Transformer blocks
            for blk_ind, blk in enumerate(self_blocks):
                # x = blk(x)
                x = blk(x, H, W)
                print(f"block{i + 1}_{blk_ind}.x.shape: {x.shape}")
            x = self_norm(x)
            print(f"norm{i + 1}.x.shape: {x.shape}")
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            print(f"Output x.shape: {x.shape}")
        # --- END shunted equiv code --- #

        # TODO: Maybe I have to return all (all stages) masks and ids?
        # mask goes to the forward_loss and ids_restore goes to the forward_decoder
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        print("---"*8, " Decoder ", "---"*8)
        # embed tokens
        x = self.decoder_embed(x)
        print(f"decoder_embed.x.shape: {x.shape}")

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        print(f"mask_tokens.shape: {mask_tokens.shape}")
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        print(f"x_.shape: {x_.shape}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        print(f"x_.shape: {x_.shape}")
        # TODO: remove this when not using cls token?
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        print(f"x.shape: {x.shape}")
        
        # add pos embed
        print(f"x + decoder_pos_embed: {x.shape} + {self.decoder_pos_embed.shape}")
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        print(f"decoder.x.shape: {x.shape}")
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        # --- START replaced code --- #
        # # target = imgs[:, :3, :, :]
        # # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        # target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5
        # loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # --- END replaced code --- #

        # --- START shunted equiv code --- #
        print("###"*8, " Loss ", "###"*8)
        print(f"mask.shape: {mask.shape}")
        print(f"pred.shape: {pred.shape}")
        target = imgs
        print(f"imgs.shape: {imgs.shape}")
        # TODO: Is it correct to use the first patch embed only?
        stage = 0
        self_patch_embed = getattr(self, f"patch_embed{stage + 1}")
        target = self.patchify(
            target, self_patch_embed.patch_size[0], self.in_c)
        print(f"patchify.target.shape: {target.shape}")
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # --- END shunted equiv code --- #

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        raise NotImplementedError("Should test different configs first.")
        return loss, pred, mask


# From PyTorch internals


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class DefaultPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(
            W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


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
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
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


def shunted_mae_vit_large_patch16_dec512d8b(**kwargs):
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


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# decoder: 512 dim, 8 blocks
shunted_mae_vit_large_patch16 = shunted_mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
