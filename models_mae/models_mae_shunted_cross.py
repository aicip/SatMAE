from typing import Union, Tuple
import random
import torch
import torch.nn as nn
from torchvision import transforms as T

from .models_mae_shunted import MaskedAutoencoderShuntedViT
from .models_mae_cross import MLP, RandomApply, default


class MaskedAutoencoderShuntedViTCross(MaskedAutoencoderShuntedViT):
    """Cross-Prediction Masked Autoencoder
    with Shunted VisionTransformer backbone"""

    def __init__(
        self,
        augment_fn1: Union[nn.Sequential, None] = None,
        augment_fn2: Union[nn.Sequential, None] = None,
        predictor_hidden_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # augmentation define
        # Adding two augments for the input to get subsampled image x1, x2: x1 = x, x2 = subsample(x)
        AUG1 = torch.nn.Sequential(T.Resize(size=(self.input_size, self.input_size)))
        AUG2 = torch.nn.Sequential(
            T.RandomResizedCrop(
                size=(self.input_size, self.input_size), scale=(0.2, 0.8)
            )
            # T.Resize(size=img_size)
        )

        self.augment1 = default(augment_fn1, AUG1)
        self.augment2 = default(augment_fn2, AUG2)
        # may need to be modified to keep size consistency
        self.predictor = MLP(
            self.decoder_embed_dim, 
            self.patch_embed1.num_patches, # type: ignore
            predictor_hidden_size
        )

    def forward_decoder(self, 
                        x: torch.Tensor, 
                        ids_restore: torch.Tensor) -> Union[torch.Tensor, 
                                                            torch.Tensor]:
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
            dec_emd = blk(x)
            if self.print_level > 1:
                print(f"\tOutputs from block_{blk_ind}: (x.shape): {x.shape})")

        # x = self.decoder_norm(dec_emd) # TODO: Removed by @maofenggg ?
        # if self.print_level > 1:
        #     print(f"norm.x.shape: {x.shape}")

        # predictor projection
        x = self.decoder_pred(dec_emd) # type: ignore
        if self.print_level > 1:
            print(f"decoder_pred.x.shape: {x.shape}")

        # remove cls token
        x = x[:, 1:, :]
        if self.print_level > 1:
            print(f"out.x.shape: {x.shape}")

        return x, dec_emd # type: ignore

    def forward(self, 
                imgs: torch.Tensor, 
                mask_seed: Union[None, int] = None, 
                consistent_mask: bool = False,
                **kwargs) -> Tuple[torch.Tensor, 
                                   torch.Tensor, 
                                   torch.Tensor]:
        img1, img2 = self.augment1(imgs), self.augment2(imgs)

        if mask_seed is None and consistent_mask:
            mask_seed = int(torch.randint(0, 100000000, (1,)).item())        
        if mask_seed is not None:
            torch.manual_seed(mask_seed)
            
        latent1, mask1, ids_restore1 = self.forward_encoder(img1)
        
        if mask_seed is not None and consistent_mask:
            torch.manual_seed(mask_seed)
            
        latent2, mask2, ids_restore2 = self.forward_encoder(img2)

        pred1, dec_emd_1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
        pred2, dec_emd_2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]

        cross_pred = self.predictor(dec_emd_2[:, 1:, :])

        if self.loss == "mse":
            loss1 = self.forward_loss_mse(img1, pred1, mask1)
            loss2 = self.forward_loss_mse(img2, pred2, mask2)
        elif self.loss == "l1":
            loss1 = self.forward_loss_l1(img1, pred1, mask1)
            loss2 = self.forward_loss_l1(img2, pred2, mask2)
        else:
            raise ValueError(f"Loss type {self.loss} not supported.")

        c_loss = nn.MSELoss()
        cross_pred = self.predictor(dec_emd_2[:, 1:, :])
        cross_loss = c_loss(dec_emd_1[:, 1:, :], cross_pred)

        loss = loss1 + loss2 + cross_loss
        if self.print_level > 1:
            raise Exception("Stopping because you set the print_level > 1.")
        return loss, pred1, mask1
