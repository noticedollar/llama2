#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Not a Contribution
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

# =============================================================================
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    Cache,
    DynamicCache,
    Qwen2Attention,
    apply_rotary_pos_emb,
    Qwen2ForCausalLM
)
from packaging import version
from importlib.metadata import version as impLib_version
from genai_lib.llm.dev.model_adaptation.common.spinquant import R3Hadamard

def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x

orig_embedding_fwd = modeling_qwen2.Qwen2RotaryEmbedding.forward
def adapted_RotaryEmbedding(self, x, position_ids, seq_len=None, *args, **kwargs):
    if isinstance(position_ids, tuple) and len(position_ids) == 2:
        return position_ids
    else:
        # transformers will move to taking in position_ids like llama moving forward for qwen2. Qwen2RotaryEmbedding -> https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
        #In transformers version > 4.44.2, the rotary_embedding's forward function returns position_embeddings (cos, sin) for each position_id in the batch.
        if version.parse(impLib_version('transformers')) > version.parse("4.44.2"):
            return orig_embedding_fwd(self, x, position_ids=position_ids, *args, **kwargs)

        cos, sin= orig_embedding_fwd(self, x, seq_len=seq_len, *args, **kwargs)
        #  For versions < 4.44.2, it returns a single view of cos, sin using the provided seq_len. Therefore, we need to stack the position embeddings for each position_id in the batch.
        cos, sin = (torch.stack([cos[[position_ids[i]]] for i in range(position_ids.shape[0])]),
                    torch.stack([sin[[position_ids[i]]] for i in range(position_ids.shape[0])]))
        return cos, sin

class QcQwen2Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super(QcQwen2Attention, self).__init__(config, layer_idx)

        if getattr(self.config, "enable_r3_hadamard", False):
            """ R3 Hadamard sourced from SpinQuant paper: https://arxiv.org/pdf/2405.16406"""
            self.q_R3 = R3Hadamard(head_dim = self.head_dim)
            self.k_R3 = R3Hadamard(head_dim = self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            if isinstance(position_ids, (tuple, list)): # QC
                position_embeddings  = position_ids
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
        query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        if getattr(self.config, "enable_r3_hadamard", False):
            query_states = self.q_R3(query_states)
            key_states = self.k_R3(key_states)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position,
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


orig_causal_mask = modeling_qwen2.Qwen2Model._update_causal_mask
def adapted_update_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return orig_causal_mask(self, attention_mask, *args, **kwargs)


def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        key_cat_dim = -1 if transposed_key_cache else -2

        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = value_cache
        return key_cache, value_cache


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False


def DynamicCache_to_legacy_cache(self):

    """
    Converts the DynamicCache instance to its equivalent in the legacy cache format for backward compatibility.

    The past_key_values passed into the model as input is a tuple.
    The LlamaModel converts it into a Cache object if it isn't one already. Within the model, past_key_values flow as a DynamicCache object.
    Just before returning the output, the LlamaModel converts the DynamicCache back to the legacy cache (tuple format).
    Since we added new attributes to our Cache object, we need to ensure they are included as additional entries in the returned tuple."""

    legacy_cache = ()
    for layer_idx in range(len(self)):
        legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
    if "anchor_buffer" in dir(self):
        return (legacy_cache, self.anchor_buffer)
    return legacy_cache


class QcQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        **loss_kwargs,
    ):
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)

        return super().forward(
            input_ids = input_ids,
            attention_mask= attention_mask,
            position_ids= position_ids,
            past_key_values= past_key_values,
            inputs_embeds= inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **loss_kwargs)
