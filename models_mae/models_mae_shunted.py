from typing import Tuple, Union
import math
import torch
import torch.nn as nn

from typing import List
from timm.models.vision_transformer import Block
from shunted import Block as ShuntedBlock
from shunted import Head as ShuntedHead
from shunted import OverlapPatchEmbed
from shunted import PatchEmbed as ShuntedPatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderShuntedViT(nn.Module):
    """Masked Autoencoder with Shunted VisionTransformer backbone"""
    # TODO: Remove print_level and print after model is 100% stable
    def __init__(
        self,
        # MAE arguments
        input_size: int = 64,
        input_channels: int = 3,
        patch_size: List[int] = [4, 4],
        dim_model: List[int] = [512, 768],
        # Encoder Parameters
        encoder_num_layers: List[int] = [8, 12],
        encoder_num_heads: List[int] =[ 8, 12],
        # Decoder paramters
        decoder_embed_dim: int = 512,
        decoder_num_layers: int = 8,
        decoder_num_heads: int = 16,
        # Feedforward parameters
        mlp_ratios: List[int] = [4, 4],
        norm_pix_loss: bool = False,
        drop_path_rate: float = 0.0,
        mask_ratio: float = 0.75,
        # shunted arguments
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        sr_ratios: List[int] = [2, 1],
        num_conv: int = 0 ,
        use_shunted_head: bool = False,
        use_overlap_patch_embed: bool = False,
        # Others
        print_level: int = 0,  # for level>1 it ony runs one pass
        norm_layer = nn.LayerNorm,
        loss: str = "mse",
        **kwargs,
    ):
        super().__init__()
        self.print_level = print_level
        if isinstance(patch_size, str):
            sep = "+"
            to_list = lambda x: [int(y) for y in x.split(sep)]
            patch_size = to_list(patch_size)
        if self.print_level > 0:
            print("--" * 8, "Config", "--" * 8)
            print(f"img_size: {input_size}")
            print(f"patch_sizes: {patch_size}")
            print(f"input_channels: {input_channels}")
            print(f"dim_model: {dim_model}")
            print(f"encoder_num_heads: {encoder_num_heads}")
            print(f"encoder_num_layers: {encoder_num_layers}")
            print(f"mlp_ratios: {mlp_ratios}")
            print(f"decoder_embed_dim: {decoder_embed_dim}")
            print(f"decoder_num_layers: {decoder_num_layers}")
            print(f"decoder_num_heads: {decoder_num_heads}")
            print(f"sr_ratios: {sr_ratios}")
            print(f"num_conv: {num_conv}")
            print(f"mask_ratio: {mask_ratio}")
            print(f"use_overlap_patch_embed: {use_overlap_patch_embed}")
            print(f"use_shunted_head: {use_shunted_head}")

            print("--" * 8, "Init Encoder", "--" * 8)
        assert (
            len(patch_size)
            == len(dim_model)
            == len(encoder_num_heads)
            == len(mlp_ratios)
            == len(encoder_num_layers)
            == len(sr_ratios)
        )
        self.input_channels = input_channels
        self.depths = encoder_num_layers
        self.num_stages = len(encoder_num_layers)
        self.use_shunted_head = use_shunted_head
        self.decoder_embed_dim = decoder_embed_dim
        self.loss = loss.lower()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.input_size = input_size
        self.patch_sizes = patch_size
        self.embed_dims = dim_model
        self.num_heads = encoder_num_heads
        self.mlp_ratios = mlp_ratios
        self.depths = encoder_num_layers
        self.mask_ratio = mask_ratio
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_num_layers))
        ]  # stochastic depth decay rule
        cur = 0
        next_embed_img_size = next_patch_H = next_patch_W = input_size
        for i in range(self.num_stages):
            if i == 0 and self.use_shunted_head:
                # This is essentially a linear+patch embedding layer accroding to the authors:
                # https://github.com/OliverRensu/Shunted-Transformer/issues/13
                patch_embed = ShuntedHead(
                    num_conv,
                    patch_size=patch_size[i],
                    stride=patch_size[i],
                    in_chans=self.input_channels,
                    embed_dim=dim_model[i],
                )
            else:
                if use_overlap_patch_embed:  # Doesn't work
                    patch_embed = OverlapPatchEmbed(
                        img_size=next_embed_img_size,
                        patch_size=patch_size[i],
                        stride=patch_size[i],
                        in_chans=self.input_channels if i == 0 else dim_model[i - 1],
                        embed_dim=dim_model[i],
                    )
                else:
                    patch_embed = ShuntedPatchEmbed(
                        img_size=next_embed_img_size,
                        patch_size=patch_size[i],
                        in_chans=self.input_channels if i == 0 else dim_model[i - 1],
                        embed_dim=dim_model[i],
                    )

            if self.print_level > 0:
                print(f"++ Stage {i+1}")
            # Find next patch embedding shape
            dummy = torch.zeros(
                1,
                self.input_channels if i == 0 else dim_model[i - 1],
                next_embed_img_size,
                next_embed_img_size,
            )
            if self.print_level > 0:
                print(f"\tPatch_embed.in.shape: {dummy.shape}")
            patch_out, next_patch_H, next_patch_W = patch_embed.forward(dummy)
            num_patches = patch_out.shape[1]
            patch_embed.num_patches = num_patches
            assert next_patch_H == next_patch_W
            if self.print_level > 0:
                print(f"\tNumber of patches: {num_patches}")
                print(f"\tPatch_embed(H, W): ({next_patch_H}, {next_patch_W})")
                print(f"\tPatch_embed.out: (1, {num_patches}, {dim_model[i]})")
            next_embed_img_size = next_patch_H
            if i == 0:
                # Size will be reduced due to the mask
                next_embed_img_size = (
                    next_embed_img_size**2 * (1 - self.mask_ratio)
                ) ** 0.5
                try:
                    assert int(next_embed_img_size) == next_embed_img_size
                except Exception as e:
                    print(
                        f"\t\t(H * W * (1 - self.mask_ratio))**0.5 = {next_embed_img_size}"
                    )
                    raise e
                next_embed_img_size = int(next_embed_img_size)
                if self.print_level > 0:
                    print(f"\tApplied mask_ratio: {self.mask_ratio}")
            if self.print_level > 0:
                print("\tNext Patch_embed.img_size: ", next_embed_img_size)

            # Encoder Transformer Block
            blocks = nn.ModuleList(
                [
                    ShuntedBlock(
                        dim=dim_model[i],
                        num_heads=encoder_num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                    )
                    for j in range(encoder_num_layers[i])
                ]
            )
            if self.print_level > 0:
                print(f"\tTransformer Block info:")
                print(f"\t\tDepth: {encoder_num_layers[i]}")
                print(f"\t\tHeads: {encoder_num_heads[i]}")
                print(f"\t\tEmbed Dim: {dim_model[i]}")
                print(f"\t\tMLP Ratio: {mlp_ratios[i]}")
                print(f"\t\tSR Ratio: {sr_ratios[i]}")
                print(
                    f"\t\tDrop Paths {[dpr[cur + j] for j in range(encoder_num_layers[i])]}"
                )
                print(f"\t\tDrop Rate {drop_rate}")
            # Norm Layer
            norm = norm_layer(dim_model[i])
            cur += encoder_num_layers[i]

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
        if self.print_level > 0:
            print("--" * 8, "Init Decoder", "--" * 8)
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(
            dim_model[-1], decoder_embed_dim, bias=True  # replaced with shunted equiv
        )
        if self.print_level > 0:
            print(f"decoder_embed.shape: {self.decoder_embed.weight.shape}")

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        if self.print_level > 0:
            print(f"mask_token.shape: {self.mask_token.shape}")

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(# +1: cls_token
                1, 
                self.patch_embed1.num_patches + 1, # type: ignore
                decoder_embed_dim  
            ),
            requires_grad=False,
        )  # fixed sin-cos embedding

        if self.print_level > 0:
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
                for _ in range(decoder_num_layers)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if self.print_level > 0:
            print(f"decoder_norm.shape: {self.decoder_norm.weight.shape}")
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size[-1] ** 2 * self.input_channels, bias=True
        )  # decoder to patch
        if self.print_level > 0:
            print(f"decoder_pred.shape: {self.decoder_pred.weight.shape}")
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self) -> None:
        if self.print_level > 0:
            print("--" * 8, "Init Weights", "--" * 8)
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
        if self.print_level > 0:
            print(f"self.patch_embed{patch_ind + 1}")
            if self.print_level > 0:
                if self.use_shunted_head:
                    print(f"\tshape={self_patch_embed.conv[0].weight.shape}")
                else:
                    print(f"\tshape={self_patch_embed.proj.weight.shape}")
                print(f"\tpatches={self_patch_embed.num_patches}")
                print(f"\tsqrt(pathes)={int(self_patch_embed.num_patches**0.5)}")
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            # replaced with shunted equiv
            int(self_patch_embed.num_patches**0.5),
            cls_token=True,
        )
        if self.print_level > 0:
            print(
                f"decoder_pos_embed({self.decoder_pos_embed.shape[-1]}, {int(self_patch_embed.num_patches**0.5)}).shape: {decoder_pos_embed.shape}"
            )
            print(f"self.decoder_pos_embed.shape: {self.decoder_pos_embed.shape}")
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # Intialize mask token
        for i in range(self.num_stages):
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            if i == 0 and self.use_shunted_head:
                w = self_patch_embed.conv[0].weight.data
            else:
                w = self_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            # torch.nn.init.normal_(getattr(self, f"cls_token{i + 1}"), std=0.02)
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
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

    def patchify(self, imgs: torch.Tensor, p: int, c: int) -> torch.Tensor:
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        if self.print_level > 2:
            print("** Patchify")
            print(f"\timgs.shape: {imgs.shape}")
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        if self.print_level > 2:
            print(f"\tc: {c}")
            print(f"\tp: {p}")
            print(f"\th: {h}, w: {w}")
            print(f"\tp**2={p**2}")
            print(f"\tp**2*c={p**2*c}")
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x: torch.Tensor, p: int, c: int) -> torch.Tensor:
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

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, 
                                                       torch.Tensor, 
                                                       torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if self.print_level > 2:
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
        if self.print_level > 2:
            print(f"\t\tids_restore.shape: {ids_restore.shape}")
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        if self.print_level > 2:
            print(f"\t\tmask.shape: {mask.shape}")

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, 
                                                       torch.Tensor, 
                                                       torch.Tensor]:
        B = x.shape[0]  # Batch Size
        if self.print_level > 1:
            print("--" * 8, " Encoder ", "--" * 8)
            print(f"Original x.shape: {x.shape}")
        for i in range(self.num_stages):  # 4 stages
            if self.print_level > 1:
                print("++", f"Stage {i+1}")
            self_patch_embed = getattr(self, f"patch_embed{i + 1}")
            # self_pos_embed = getattr(self, f"pos_embed{i + 1}")
            # self_cls_token = getattr(self, f"cls_token{i + 1}")
            self_blocks = getattr(self, f"blocks{i + 1}")
            self_norm = getattr(self, f"norm{i + 1}")

            # embed patches
            if (i != 0 or not self.use_shunted_head) and self.print_level > 1:
                print(
                    f"\tExpected patch_embed{i+1}.img_size: {self_patch_embed.img_size}"
                )
            x, H, W = self_patch_embed(x)  # x.shape: (1, patches, embed_dim)
            if self.print_level > 1:
                # (1, 1024, 64), (1, 256, 128), (1, 64, 256), (1, 16, 512)
                print(f"\tpatch_embed{i + 1}.x.shape: {x.shape}")
                # B, H, W: (1, 32, 32), (1, 16, 16), (1, 8, 8), (1, 4, 4)
                print(f"\t\tB, H, W: {B, H, W}")
            assert H == W

            # add pos embed w/o cls token
            # print(f"\tExpected pos_embed{i + 1}.shape: {self_pos_embed[:, 1:, :].shape}")
            # x = x + self_pos_embed[:, 1:, :]
            # print(f"\tpos_embed.x.shape: {x.shape}")

            # masking: length -> length * mask_ratio
            if i == 0:
                x, mask, ids_restore = self.random_masking(x)
                if self.print_level > 1:
                    print(f"\trandom_masking{i + 1}.x.shape: {x.shape}")
                H = (H * W * (1 - self.mask_ratio)) ** 0.5
                try:
                    assert int(H) == H
                except Exception as e:
                    print(f"\t\t(H * W * (1 - self.mask_ratio))**0.5 = {H}")
                    raise e
                H = W = int(H)
                if self.print_level > 1:
                    print(f"\t\tModified H, W: {H, W}")

            # append cls token
            # cls_token = self_cls_token + self_pos_embed[:, :1, :]
            # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            # x = torch.cat((cls_tokens, x), dim=1)
            # print(f"cls_token{i + 1}.x.shape: {x.shape}")

            # apply Transformer blocks
            if self.print_level > 1:
                print("\t** Transformer blocks")
            for blk_ind, blk in enumerate(self_blocks):
                if self.print_level > 1:
                    print(
                        f"\t\t{x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous().shape}"
                    )
                x = blk(x, H, W)
                if self.print_level > 1:
                    print(f"\t\tOutputs from block{i + 1}_{blk_ind}.x.shape: {x.shape}")

            # apply normalization
            x = self_norm(x)
            if self.print_level > 1:
                print(f"\tnorm{i + 1}.x.shape: {x.shape}")
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if self.print_level > 1:
                print(f"\tOutput x.shape: {x.shape}")
                if i == self.num_stages - 1:
                    print(
                        f"\tOutput x.contiguous().shape: {x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous().shape}"
                    )

        return x, mask, ids_restore # type: ignore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        if self.print_level > 1:
            print("--" * 8, " Decoder ", "--" * 8)
            print(f"In x.shape: {x.shape}")
            print(f"In ids_restore.shape: {ids_restore.shape}")
        # embed tokens
        x = self.decoder_embed(x)
        if self.print_level > 1:
            print(f"decoder_embed.x.shape: {x.shape}")

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        if self.print_level > 2:
            print(f"mask_tokens.shape: {mask_tokens.shape}")
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        if self.print_level > 2:
            print(f"x_.cat(mask_tokens).shape: {x_.shape}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        if self.print_level > 2:
            print(f"x_.gather(ids_restore).shape: {x_.shape}")

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        if self.print_level > 1:
            print(f"x.cat(x_).shape: {x.shape}")

        # add pos embed
        if self.print_level > 1:
            print(f"x + decoder_pos_embed: {x.shape} + {self.decoder_pos_embed.shape}")
        x = x + self.decoder_pos_embed
        if self.print_level > 1:
            print(f"decoder_pos_embed+x.shape: {x.shape}")

        # apply Transformer blocks
        if self.print_level > 1:
            print("** Transformer blocks")
        for blk_ind, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            if self.print_level > 1:
                print(f"\tOutputs from block_{blk_ind}: (x.shape): {x.shape})")
        x = self.decoder_norm(x)
        if self.print_level > 1:
            print(f"norm.x.shape: {x.shape}")

        # predictor projection
        x = self.decoder_pred(x)
        if self.print_level > 1:
            print(f"decoder_pred.x.shape: {x.shape}")

        # remove cls token
        x = x[:, 1:, :]
        if self.print_level > 1:
            print(f"out.x.shape: {x.shape}")

        return x

    def forward_loss_mse(self, 
                         imgs: torch.Tensor, 
                         pred: torch.Tensor, 
                         mask: Union[None, torch.Tensor]) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.print_level > 1:
            print("--" * 8, " Loss ", "--" * 8)
            if mask is not None:
                print(f"In mask.shape: {mask.shape}")
            print(f"In pred.shape: {pred.shape}")
        target = imgs
        if self.print_level > 1:
            print(f"In imgs.shape: {imgs.shape}")

        stage = self.num_stages - 1
        self_patch_embed = getattr(self, f"patch_embed{stage + 1}")
        patch_size = self_patch_embed.patch_size[0]
        target = self.patchify(target, patch_size, self.input_channels)
        if self.print_level > 1:
            print(f"patchify.target.shape: {target.shape}")
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if self.print_level > 1:
            print(f"(pred-target).shape: ({pred.shape}-{target.shape})")
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()

        return loss

    def forward_loss_l1(self, 
                        imgs: torch.Tensor, 
                        pred: torch.Tensor, 
                        mask: Union[None, torch.Tensor]) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        if self.print_level > 1:
            print("--" * 8, " Loss ", "--" * 8)
            if mask is not None:
                print(f"In mask.shape: {mask.shape}")
            print(f"In pred.shape: {pred.shape}")
        target = imgs
        if self.print_level > 1:
            print(f"In imgs.shape: {imgs.shape}")
        stage = self.num_stages - 1
        self_patch_embed = getattr(self, f"patch_embed{stage + 1}")
        patch_size = self_patch_embed.patch_size[0]
        target = self.patchify(target, patch_size, self.input_channels)
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
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def forward(self, 
                imgs: torch.Tensor, 
                mask_seed: Union[None, int] = None, 
                **kwargs) -> Tuple[torch.Tensor, 
                                   torch.Tensor, 
                                   torch.Tensor]:
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
            
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        loss_full = self.loss.endswith("full")
        loss_mask = None if loss_full else mask
        if self.loss.startswith("mse"):
            loss = self.forward_loss_mse(imgs, pred, mask=loss_mask)
        elif self.loss.startswith("l1"):
            loss = self.forward_loss_l1(imgs, pred, mask=loss_mask)
        else:
            raise ValueError(f"Loss type {self.loss} not supported.")

        if self.print_level > 1:
            raise Exception("Stopping because you set the print_level > 1.")
        return loss, pred, mask
