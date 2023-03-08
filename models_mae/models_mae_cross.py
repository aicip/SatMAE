import random
import torch
import torch.nn as nn

from torchvision import transforms as T
from .models_mae import MaskedAutoencoderViT

# xformers._is_functorch_available = True


# adding two function, MLP is for prediction, RandomApply is for augment


def MLP(emd_dim, channel=64, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(emd_dim, hidden_size),
        nn.BatchNorm1d(channel),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, emd_dim),
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


class MaskedAutoencoderViTCross(MaskedAutoencoderViT):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        augment_fn1=None,
        augment_fn2=None,
        predictor_hidden_size=2048,
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
            self.decoder_embed_dim, self.num_patches, predictor_hidden_size
        )

    def forward(self, imgs, mask_ratio=0.75):
        img1, img2 = self.augment1(imgs), self.augment2(imgs)

        latent1, mask1, ids_restore1 = self.forward_encoder(img1, mask_ratio)
        latent2, mask2, ids_restore2 = self.forward_encoder(img2, mask_ratio)

        pred1, dec_emd_1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
        pred2, dec_emd_2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]

        # latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        # TODO: Add flag for loss function
        if self.loss == "mse":
            loss1 = self.forward_loss_mse(img1, pred1, mask1)
            loss2 = self.forward_loss_mse(img2, pred2, mask2)
            # cross_loss = self.forward_loss_mse(dec_emd_1[:, 1:, :], cross_pred)
        elif self.loss == "l1":
            loss1 = self.forward_loss_l1(img1, pred1, mask1)
            loss2 = self.forward_loss_l1(img2, pred2, mask2)
            # cross_loss = self.forward_loss_l1(dec_emd_1[:, 1:, :], cross_pred)
        else:
            raise ValueError(f"Loss type {self.loss} not supported.")

        c_loss = nn.MSELoss()
        cross_pred = self.predictor(dec_emd_2[:, 1:, :])
        cross_loss = c_loss(dec_emd_1[:, 1:, :], cross_pred)

        loss = loss1 + loss2 + cross_loss

        return loss, pred1, mask1
