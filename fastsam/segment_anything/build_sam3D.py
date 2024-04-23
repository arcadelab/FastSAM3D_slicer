# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D, ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import torch
# def build_sam3D_vit_h(checkpoint=None):
#     return _build_sam3D(
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[7, 15, 23, 31],
#         checkpoint=checkpoint,
#     )


# build_sam3D = build_sam3D_vit_h


# def build_sam3D_vit_l(checkpoint=None):
#     return _build_sam3D(
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[5, 11, 17, 23],
#         checkpoint=checkpoint,
#     )


# def build_sam3D_vit_b(checkpoint=None):
#     return _build_sam3D(
#         # encoder_embed_dim=768,
#         encoder_embed_dim=384,
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_global_attn_indexes=[2, 5, 8, 11],
#         checkpoint=checkpoint,
#     )

def build_FastSAM3D(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=6,
        encoder_global_attn_indexes=[2,5,8,11],
        window_size = 0,
        skip_layer = 2,
        checkpoint=checkpoint,
    )

def build_SAMMed3D(checkpoint=None):
    return _build_sam3D(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        window_size = 14,
        skip_layer = 0,
        checkpoint=checkpoint,
    )

def build_MedSAM(checkpoint = None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size = 1024,
        checkpoint=checkpoint,
        encoder_adapter= False,
    )
def build_SAMMed2D(checkpoint = None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size = 256,
        checkpoint=checkpoint,
        encoder_adapter= True,
    )
sam_model_registry3D = {
    "MedSAM": build_MedSAM,
    "FastSAM3D": build_FastSAM3D,
    "SAMMed2D": build_SAMMed2D,
    "SAMMed3D":build_SAMMed3D,
}

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos = True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train = encoder_adapter,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=device)
            if 'model' in state_dict.keys():
                sam.load_state_dict(state_dict['model'])
            else:
                sam.load_state_dict(state_dict)
    return sam

def _build_sam3D(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    window_size,
    skip_layer,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=window_size,
            out_chans=prompt_embed_dim,
            skip_layer = skip_layer,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if checkpoint is not None:
        model_dict = torch.load(checkpoint,map_location = device)
        state_dict = model_dict['model_state_dict']
        sam.load_state_dict(state_dict)
    return sam