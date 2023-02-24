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
                 # MAE arguments
                 img_size=224,
                 patch_sizes=[16, 16, 16, 16],
                 in_chans=3,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
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
                 num_conv=0,
                 mask_ratio=0.75
                 ):
        super().__init__()
        print("--"*8, "Config", "--"*8)
        print(f"img_size: {img_size}")
        print(f"patch_sizes: {patch_sizes}")
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
        print(f"mask_ratio: {mask_ratio}")
        
        print("--"*8, "Init Encoder", "--"*8)
        self.in_c = in_chans
        self.depths = depths
        self.num_stages = num_stages
        self.mask_ratio = mask_ratio

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        next_embed_img_size = next_patch_H = next_patch_W = img_size
        self.used_shunted_head = False # TODO: Remove after finished testing
        for i in range(num_stages):
            # if i == 0 and False: # TODO: Decide if we want to use it
            #     # This is essentially a linear+patch embedding layer. 
            #     # https://github.com/OliverRensu/Shunted-Transformer/issues/13
            #     patch_embed = ShuntedHead(num_conv, 
            #                               patch_size=patch_sizes[i], 
            #                               in_chans=in_chans)
            #     self.used_shunted_head = True # TODO: Remove after finished testing
            # else:  # TODO: Decide classic or overlap patch embedding
            patch_embed = DefaultPatchEmbed(img_size=next_embed_img_size,
                                            patch_size=patch_sizes[i],
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            # patch_embed = OverlapPatchEmbed(img_size=next_embed_img_size,
            #                                 patch_size=patch_sizes[i],
            #                                 stride=patch_sizes[i],
            #                                 in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                                 embed_dim=embed_dims[i])
            print(f"++ Stage {i+1}")        
            # Find next patch embedding shape
            dummy = torch.zeros(1, 
                                in_chans if i == 0 else embed_dims[i - 1], 
                                next_embed_img_size, 
                                next_embed_img_size)
            print(f"\tPatch_embed.in.shape: {dummy.shape}")
            _, next_patch_H, next_patch_W = patch_embed.forward(dummy)
            assert next_patch_H == next_patch_W
            print(f"\tNumber of patches: {patch_embed.num_patches}")
            print(f"\tPatch_embed(H, W): ({next_patch_H}, {next_patch_W})")
            print(f"\tPatch_embed.out: (1, {patch_embed.num_patches}, {embed_dims[i]})")
            next_embed_img_size = next_patch_H
            if i == 0:
                # Size will be reduced due to the mask                
                next_embed_img_size = (next_embed_img_size**2 * (1-self.mask_ratio))**0.5
                try:
                    assert int(next_embed_img_size) == next_embed_img_size
                except Exception as e:
                    print(f"\t\t(H * W * (1 - self.mask_ratio))**0.5 = {next_embed_img_size}")
                    raise e
                next_embed_img_size = int(next_embed_img_size)
                print(f"\tApplied mask_ratio: {self.mask_ratio}")
            print("\tNext Patch_embed.img_size: ", next_embed_img_size)
            
            # Encoder Transformer Block
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
            print(f"\tTransformer Block info:")
            print(f"\t\tDepth: {depths[i]}")
            print(f"\t\tHeads: {num_heads[i]}")
            print(f"\t\tEmbed Dim: {embed_dims[i]}")
            print(f"\t\tMLP Ratio: {mlp_ratios[i]}")
            print(f"\t\tSR Ratio: {sr_ratios[i]}")
            print(f"\t\tDrop Paths {[dpr[cur + j] for j in range(depths[i])]}")
            print(f"\t\tDrop Rate {drop_rate}")
            # Norm Layer
            norm = norm_layer(embed_dims[i])
            cur += depths[i]            

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"blocks{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
        # Note: Replaced the orignal patch_embed, block, norm (self vars)
        # With the self vars:
        # patch_embed1, patch_embed2, patch_embed3, patch_embed4
        # block1, block2, block3, block4
        # norm1, norm2, norm3, norm4

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        print("--"*8, "Init Decoder", "--"*8)
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dims[-1],  # replaced with shunted equiv
                                       decoder_embed_dim,
                                       bias=True)
        print(f"decoder_embed.shape: {self.decoder_embed.weight.shape}")

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        print(f"mask_token.shape: {self.mask_token.shape}")

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1,
                                                          self.patch_embed1.num_patches+1, # +1: cls_token
                                                          decoder_embed_dim),
                                              requires_grad=False
                                              )  # fixed sin-cos embedding
        print(1, self.patch_embed1.num_patches+1,decoder_embed_dim)
        print(f"decoder_pos_embed.shape: {self.decoder_pos_embed.shape}")
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
        print(f"decoder_norm.shape: {self.decoder_norm.weight.shape}")
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                      patch_sizes[-1]**2 * in_chans,
                                      bias=True)  # decoder to patch
        print(f"decoder_pred.shape: {self.decoder_pred.weight.shape}")
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        print("--"*8, "Init Weights", "--"*8)
        # initialization

        # Intialize Encoder pos embed
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

        # Intialize Decoder pos and patch embed
        # patch_ind = self.num_stages - 1
        patch_ind = 0
        self_patch_embed = getattr(self, f"patch_embed{patch_ind + 1}")
        print(f"self.patch_embed{patch_ind + 1}")
        print(f"\tshape={self_patch_embed.proj.weight.shape}")
        print(f"\tpatches={self_patch_embed.num_patches}")
        print(f"\tsqrt(pathes)={int(self_patch_embed.num_patches**0.5)}")
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            # replaced with shunted equiv
            int(self_patch_embed.num_patches**0.5),
            cls_token=True,
        )
        print(f"decoder_pos_embed({self.decoder_pos_embed.shape[-1]}, {int(self_patch_embed.num_patches**0.5)}).shape: {decoder_pos_embed.shape}")
        print(f"self.decoder_pos_embed.shape: {self.decoder_pos_embed.shape}")
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # Intialize mask token
        for i in range(self.num_stages):
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            if i == 0 and self.used_shunted_head: # TODO: Remove after finished testing
                w = self_patch_embed.conv[0].weight.data
            else:
                w = self_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            # torch.nn.init.normal_(getattr(self, f"cls_token{i + 1}"), std=0.02)
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
        print("** Patchify")
        # p = self.patch_embed.patch_size[0]
        print(f"\timgs.shape: {imgs.shape}")
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        print(f"\tc: {c}")
        print(f"\tp: {p}")
        print(f"\th: {h}, w: {w}")
        print(f"\tp**2={p**2}")
        print(f"\tp**2*c={p**2*c}")
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

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        print("\t** Random masking")
        print(f"\t\tmask_ratio: {self.mask_ratio}")
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        print(f"\t\tids_restore.shape: {ids_restore.shape}")
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print(f"\t\tmask.shape: {mask.shape}")

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        B = x.shape[0] # This should be 1
        print("--"*8, " Encoder ", "--"*8)
        print(f"Original x.shape: {x.shape}")
        for i in range(self.num_stages): # 4 stages
            print("++", f"Stage {i+1}")
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            # self_pos_embed = getattr(self, f"pos_embed{i + 1}")
            # self_cls_token = getattr(self, f"cls_token{i + 1}")
            self_blocks = getattr(self, f"blocks{i + 1}")
            self_norm = getattr(self, f"norm{i + 1}")

            # embed patches
            if i != 0 or not self.used_shunted_head:
                print(f"\tExpected patch_embed{i+1}.img_size: {self_patch_embed.img_size}")
            x, H, W = self_patch_embed(x) # x.shape: (1, patches, embed_dim)
            print(f"\tpatch_embed{i + 1}.x.shape: {x.shape}") # (1, 1024, 64), (1, 256, 128), (1, 64, 256), (1, 16, 512)
            print(f"\t\tB, H, W: {B, H, W}") # B, H, W: (1, 32, 32), (1, 16, 16), (1, 8, 8), (1, 4, 4)
            assert H == W
            
            # add pos embed w/o cls token
            # print(f"\tExpected pos_embed{i + 1}.shape: {self_pos_embed[:, 1:, :].shape}")
            # x = x + self_pos_embed[:, 1:, :]
            # print(f"\tpos_embed.x.shape: {x.shape}")

            # masking: length -> length * mask_ratio
            if i==0:
                x, mask, ids_restore = self.random_masking(x)
                print(f"\trandom_masking{i + 1}.x.shape: {x.shape}")
                H = (H * W * (1 - self.mask_ratio))**0.5
                try:
                    assert int(H) == H
                except Exception as e:
                    print(f"\t\t(H * W * (1 - self.mask_ratio))**0.5 = {H}")
                    raise e
                H = W = int(H)
                print(f"\t\tModified H, W: {H, W}")

        
            # append cls token
            # cls_token = self_cls_token + self_pos_embed[:, :1, :]
            # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            # x = torch.cat((cls_tokens, x), dim=1)
            # print(f"cls_token{i + 1}.x.shape: {x.shape}")

            # apply Transformer blocks
            print("\t** Transformer blocks")
            for blk_ind, blk in enumerate(self_blocks):
                print(f"\t\t{x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous().shape}")
                x = blk(x, H, W)
                print(f"\t\tOutputs from block{i + 1}_{blk_ind}.x.shape: {x.shape}")

            # apply normalization
            x = self_norm(x)
            print(f"\tnorm{i + 1}.x.shape: {x.shape}")
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            print(f"\tOutput x.shape: {x.shape}")

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        print("--"*8, " Decoder ", "--"*8)
        print(f"In x.shape: {x.shape}")
        print(f"In ids_restore.shape: {ids_restore.shape}")
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
        print(f"x_.cat(mask_tokens).shape: {x_.shape}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        print(f"x_.gather(ids_restore).shape: {x_.shape}")
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        print(f"x.cat(x_).shape: {x.shape}")
        
        # add pos embed
        print(f"x + decoder_pos_embed: {x.shape} + {self.decoder_pos_embed.shape}")
        x = x + self.decoder_pos_embed
        print(f"decoder_pos_embed+x.shape: {x.shape}")

        # apply Transformer blocks
        print("** Transformer blocks")
        for blk_ind, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            print(f"\tOutputs from block_{blk_ind}: (x.shape): {x.shape})")
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
        print("--"*8, " Loss ", "--"*8)
        print(f"In mask.shape: {mask.shape}")
        print(f"In pred.shape: {pred.shape}")
        target = imgs
        print(f"In imgs.shape: {imgs.shape}")
        
        stage = self.num_stages - 1
        self_patch_embed = getattr(self, f"patch_embed{stage + 1}")
        target = self.patchify(
            target, self_patch_embed.patch_size[0], self.in_c)
        print(f"patchify.target.shape: {target.shape}")
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        
        print(f"(pred-target).shape: ({pred.shape}-{target.shape})")
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
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
        # decoder_embed_dim=512,
        # decoder_depth=8,
        # decoder_num_heads=16,
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
