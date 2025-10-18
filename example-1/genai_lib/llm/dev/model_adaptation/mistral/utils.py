#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities that the pipeline would need to work with the adaptations made to Mistral model. """

import torch
import functools

def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg = -100.0):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. model_id_or_path: Model name or path to pretrained model
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    '''
    mistral_model = _get_model(model_id_or_path)
    cache_position = torch.arange(model_context_len-max_input_tokens, model_context_len, device = input_tensor.device)
    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device = input_tensor.device)
    prepared_attention_mask = mistral_model._update_causal_mask(attention_mask=prepared_1d_attn_mask, input_tensor=input_embeds, output_attentions=True,
                              cache_position=cache_position, past_key_values=None, use_cache=False)
    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)
    return prepared_attention_mask

def llm_create_position_embeddings(config, position_ids=None):
    '''
    This function creates position embedding (RoPE) from the position ids.
    params:
    1. config: model configuration to create the LLamaRotaryEmbedding object, expect config for backward compatibility, in future transformers, we only expect to pass the config, the one we have in docker, takes in the req argument
    2. position_ids: required position ids passed into the model
    '''

    max_position_embeddings = config.max_position_embeddings
    rope_theta = config.rope_theta
    dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(dim=dim, max_position_embeddings=max_position_embeddings, rope_theta=rope_theta, device=device)
    cos, sin = rotary_emb(x, position_ids=position_ids)
    cos, sin = cos.unsqueeze(dim = 1), sin.unsqueeze(dim = 1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:, :dim//2]
    return cos, sin

@functools.cache
def _get_rotary_embedding(dim, max_position_embeddings, rope_theta, device):
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
    rotary_emb = MistralRotaryEmbedding(dim=dim, max_position_embeddings=max_position_embeddings, base=rope_theta, device=device)
    return rotary_emb

@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.mistral.modeling_mistral import MistralModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = 1
    model = MistralModel(config)
    return model
