# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ShuntedTransformer: https://github.com/OliverRensu/Shunted-Transformer
# --------------------------------------------------------

from functools import partial
import random
import math
import torch
import torch.nn as nn
from xformers.factory import xFormer, xFormerConfig
from timm.models.vision_transformer import Block, PatchEmbed
from torchvision import transforms as T
from xformers.factory import xFormer, xFormerConfig

from shunted import Block as ShuntedBlock
from shunted import Head as ShuntedHead
from shunted import OverlapPatchEmbed
from shunted import PatchEmbed as ShuntedPatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from 
# xformers._is_functorch_available = True



# adding two function, MLP is for prediction, RandomApply is for augment


def MLP(emd_dim, channel=64, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(emd_dim, hidden_size),
        nn.BatchNorm1d(channel),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, emd_dim)
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


