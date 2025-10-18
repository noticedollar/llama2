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
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

""" Eaglet speculative decoding algorithm """

import math
from typing import List, Optional, Tuple, Union
import aimet_torch.v2.nn.modules.custom as aimet_ops
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, LlamaAttention, LlamaDecoderLayer
from genai_lib.llm.dev.model_adaptation.llama.adaptation import QcLlamaAttention
from genai_lib.common.dev.utils import filter_outputs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)


class EagletDraftModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`EagletDecoderLayer`]
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.config = config
        self.gradient_checkpointing = False
        self.training = False
        self.layers = nn.ModuleList(
            [EagletDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=config.fc_bias)
        self.down_proj = nn.Linear(config.hidden_size, config.downsample_size, bias=False)
        self.up_proj = nn.Linear(config.downsample_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.merged_head = nn.Linear(config.downsample_size, getattr(config, "trimmed_vocabulary_length", config.vocab_size), bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.embedding_concat = aimet_ops.Concat(axis=-1)

        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference))

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward('Draft model forward')
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        hidden_states: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        cache_index: Optional[torch.Tensor]=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcLlamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        concat_embeddings = self.embedding_concat(inputs_embeds, hidden_states)
        hidden_states = self.fc(concat_embeddings)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # hidden_up
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act(hidden_states)

        # add hidden states from the down proj
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.merged_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            outputs = tuple(v for v in [logits, None, next_cache, all_hidden_states, all_self_attns] if v is not None)
            if hasattr(self.config, "output_index_filter"):
                outputs = filter_outputs(outputs, self.config.output_index_filter)
            return outputs

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        raise NotImplementedError(f"EagletDraftModel._update_causal_mask must be implemented before being invoked")


class EagletDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = QcLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        if layer_idx > 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        if hasattr(self, "input_layernorm"):
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs