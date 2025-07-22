# --------------------------------------------------------
# The following code is based on 2 codebases:
## (1) A-ViT
# https://github.com/NVlabs/A-ViT
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
## (2) DeiT
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# The code is modified to accomodate ViT training
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


## ---- ViT-T/S baseline without ETF Projector --- ##

@register_model
def vit_tiny(pretrained=False, **kwargs):

    from timm.models.etf_vision_transformer import VisionTransformer
    #from etf_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16, output_dim=0, hidden_mlp=0,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_small(pretrained=False, **kwargs):

    from timm.models.etf_vision_transformer import VisionTransformer
    #from etf_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16, output_dim=0, hidden_mlp=0,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model




## ---- ViT-T/S with ETF Projector (Ours) --- ##

#ViT-Tiny+ETF Projector: output_dim=192, hidden_mlp=768
@register_model
def evit_tiny(pretrained=False, **kwargs):

    from timm.models.etf_vision_transformer import VisionTransformer
    #from etf_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16, output_dim=192, hidden_mlp=768,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

#ViT-Small+ETF Projector: output_dim=384, hidden_mlp=1536
@register_model
def evit_small(pretrained=False, **kwargs):

    from timm.models.etf_vision_transformer import VisionTransformer
    #from etf_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16, output_dim=384, hidden_mlp=1536,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
