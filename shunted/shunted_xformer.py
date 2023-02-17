import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from dataclasses import dataclass
from xformers.components.attention import Attention, AttentionConfig, register_attention
from typing import Optional


@dataclass
class ShuntedSelfAttentionConfig(AttentionConfig):
    """
    dim                     Dimension of the input.
    num_heads               Number of heads. Default: 8
    qkv_bias                If True, add a learnable bias to q, k, v. Default: False
    qk_scale                Override default qk scale of head_dim ** -0.5 if set
    attn_drop               Dropout ratio of attention weight. Default: 0.0
    proj_drop               Dropout ratio of output. Default: 0.0
    sr_ratio                The ratio of spatial reduction of Spatial Reduction Attention. Default: 1
    """

    dim: int
    num_heads: Optional[int]
    qkv_bias: Optional[bool]
    qk_scale: Optional[float]
    attn_drop: Optional[float]
    proj_drop: Optional[float]
    sr_ratio: Optional[int]


@register_attention("shunted", ShuntedSelfAttentionConfig)
class ShuntedAttention(Attention):
    def __init__(self,
                 dim: int,
                 num_heads: Optional[int] = 8,
                 qkv_bias: Optional[bool] = False,
                 qk_scale: Optional[float] = None,
                 attn_drop: Optional[float] = 0.0,
                 proj_drop: Optional[float] = 0.0,
                 sr_ratio: Optional[int] = 1,
                 *args,
                 **kwargs,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(
                dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(
                dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(
                self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(
                self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads //
                                        2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads //
                                        2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # B head N C
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q[:, :self.num_heads//2] @
                     k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                       transpose(1, 2).view(B, C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2,
                                       C // self.num_heads, -1).transpose(-1, -2)
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
            attn2 = (q[:, self.num_heads // 2:] @
                     k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                       transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2,
                                       C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

            x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                                                              transpose(1, 2).view(B, C, H, W)).view(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Example for how to register a custom module
# @dataclass
# class NystromSelfAttentionConfig(AttentionConfig):
#     ...

# @register_attention("nystrom", NystromSelfAttentionConfig)
# class NystromAttention(Attention):
#     def __init__(
#         self,
#         dropout: float,
#         num_heads: int,
#         num_landmarks: int = 64,
#         landmark_pooling: Optional[nn.Module] = None,
#         causal: bool = False,
#         use_razavi_pinverse: bool = True,
#         pinverse_original_init: bool = False,
#         inv_iterations: int = 6,  # recommended default in paper was 6.
#         v_skip_connection: Optional[nn.Module] = None,
#         conv_kernel_size: Optional[int] = None,
#         *args,
#         **kwargs,
#     ):
#         ...


#     def forward(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
#     ):
#         ...
