#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities for the pipeline to work with static shape requirements for inputs that go into the model """

import torch
from genai_lib.llm.utils import _shift, _concat

def llm_slice_inputs_for_inference(max_input_tokens, model_context_len, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, past_seen_tokens=None, hidden_states=None):
    """
    This function is responsible for slicing the inputs based on the AR and yield them to the user.
    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. model_context_len: maximum number of tokens that the model can consume in total
    3. input_ids: input ids sent to the model
    4. inputs_embeds: input embeds sent to the model
    5. attention_mask: attention mask sent to the model
    6. position_ids: position ids sent to the model
    7. hidden_states: hidden states sent to the model

    Note: To be able to ingest all the model_context_len tokens, we need to slice using the left padding and chunk the input into chunks of max_input_tokens
    """

    input_count = 0
    for input in (input_ids, inputs_embeds):
        if input is not None:
            input_count = input_count + 1

    assert input_count == 1, "Should pass either input ids or input embeddings, not both"

    if input_ids is not None:
        input_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device
    else:
        input_length = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, input_length), dtype = torch.long, device = device)

    if position_ids is None:
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        if past_seen_tokens is not None:
            position_ids += past_seen_tokens

    for idx in range(0, input_length, max_input_tokens)[::-1]:
        idx = input_length - idx
        output_slice = {
            'attn_mask_slice':attention_mask[:, max(0, idx-max_input_tokens):idx],
            'position_ids_slice': position_ids[:, max(0, idx-max_input_tokens):idx],
        }

        if input_ids is not None:
            output_slice['input_ids_slice'] = input_ids[:, max(0, idx-max_input_tokens):idx]
        else:
            output_slice['inputs_embeds_slice'] = inputs_embeds[:, max(0, idx-max_input_tokens):idx, :]

        if hidden_states is not None:
            output_slice['hidden_states_slice'] = hidden_states[:, max(0, idx - max_input_tokens):idx]

        yield output_slice


def llm_pad_inputs(max_input_tokens, input_ids_slice=None, inputs_embeds_slice=None, pad_token=0, pad_embeds=None, pad_to_left=True):
    '''
    This function pads the input_ids/ inputs_embeds since slice may return input_ids/ inputs_embeds that is smaller in length
    than what the model accepts (AR len)

    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. input_ids_slice: the current input ids slice that is passed into the model in the current invocation
    3. inputs_embeds_slice: the current input embeds slice that is passed into the model in the current invocation
    4. pad_token: padding token, this is defaulted to 0 to avoid impacting the range of values in the activation tensor
    5. pad_embeds: Tensor with which we pad the inputs_embeds_slice. This is optional and will be used if provided.
        If this is not provided, and we are working with input embeddings, the inputs_embeds_slice tensor will be padded
        with zero and not the pad_token. The reason for this is that the pad_token could be a large non-zero value which
        will impact the range of values in the padded tensor.
    6. pad_to_left: boolean value indicating whether padding is done towards the left or right.

    '''
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    device = input.device
    input_length = input.shape[1]
    batch_size = input.shape[0]
    shape = (batch_size, max_input_tokens - input_length)

    if pad_embeds is None:
        if inputs_embeds_slice is not None:
            shape += (input.shape[-1],)
            pad_token = 0

        input_extensions = torch.full(
            shape,
            fill_value=pad_token,
            dtype=input.dtype,
            device=device
        )
    else:
        assert input.shape[-1] == pad_embeds.shape[-1]
        # we only want to extract the embeddings dimension from the passed pad_embeddings
        pad_embeds = pad_embeds[-1]
        input_extensions = pad_embeds.view(1, 1, -1).repeat(batch_size, max_input_tokens - input_length, 1).to(dtype=input.dtype, device=device)

    # left padding
    if pad_to_left:
        input = torch.cat((input_extensions, input), dim=1)
    # right padding
    else:
        input = torch.cat((input, input_extensions), dim=1)

    return input

def llm_pad_hidden_states(max_input_tokens: int, hidden_states_slice: torch.Tensor, pad_token=0, pad_to_left=True):
    '''
    This function pads the hidden states since slice may return hidden states that is smaller in length
    than what the model accepts (AR len)

    params:
    1. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    2. hidden_states_slice: the current hidden state slice that is passed into the model in the current invocation
    3. pad_token: padding token, this is defaulted to 0 to avoid impacting the range of values in the activation tensor
    4. pad_to_left: boolean value indicating whether padding is done towards the left or right.

    '''
    pad_shape = list(hidden_states_slice.shape)
    pad_shape[1] = max_input_tokens - hidden_states_slice.shape[1]
    pad = torch.full(
        pad_shape,
        fill_value=pad_token,
        dtype=hidden_states_slice.dtype,
        device=hidden_states_slice.device
    )

    # left padding
    if pad_to_left:
        padded_hidden_states_slice = torch.cat((pad, hidden_states_slice), dim=1)
    # right padding
    else:
        padded_hidden_states_slice = torch.cat((hidden_states_slice, pad), dim=1)

    return padded_hidden_states_slice

def llm_pad_input_attn_mask(attn_mask_slice, max_input_tokens, pad_to_left=True):
    """
    This function pads the 1d attention mask to make it of shape (batch_size, max_input_tokens),

    A: padded current input (0s)
    B: current valid input (1s)

    If the pad_to_left argument is set to True, it means we perform left_padding & produce the attention mask as A|B
    else, pad_to_left=False means we do right padding, & produce the attention mask as B|A

    params:
    attn_mask_slice: the attention mask which corresponds to the current slice of inputs
    max_input_tokens: the maximum tokens that can be sent to the model, in our context represents the AR length
    pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    batch_size = attn_mask_slice.shape[0]
    input_padding_length = max_input_tokens - attn_mask_slice.shape[1]

    padded_input_attn_mask = torch.zeros((batch_size, input_padding_length), dtype=torch.long,
                                         device=attn_mask_slice.device)

    #left padding
    if pad_to_left:
        attention_mask = torch.cat((
            padded_input_attn_mask,
            attn_mask_slice
        ),
            dim=1
        )
    # right padding
    else:
        attention_mask = torch.cat((
            attn_mask_slice,
            padded_input_attn_mask
        ),
            dim=1
        )

    return attention_mask


def llm_create_kv_attn_mask(unpadded_past_kv, model_context_len, max_input_tokens, batch_size, device, pad_to_left=True):
    """
    This function prepares the 1d attention mask based on the useful past key values seen so far.
    This can be visualized into two sections.
    A | B
    A: padded past kv length (0s)
    B: useful past kv length (1s)

    params:
    unpadded_past_kv: this is the useful accumulated past kv
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    model_context_len: maximum number of tokens that the model can consume in total
    batch_size: batch size of current input
    device: device to place the attention mask
    pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """

    useful_past_kv_length = unpadded_past_kv[0][1].shape[-2] if unpadded_past_kv else 0
    padded_kv_length = (model_context_len - max_input_tokens) - useful_past_kv_length

    useful_past_kv_attn_mask = torch.ones((batch_size, useful_past_kv_length), dtype=torch.long,
                                          device=device)
    padded_kv_attn_mask = torch.zeros((batch_size, padded_kv_length), dtype=torch.long, device=device)

    # left padding
    if pad_to_left:
        attention_mask = torch.cat((
            padded_kv_attn_mask,
            useful_past_kv_attn_mask
        ),
            dim=1
        )
    #right padding
    else:
        attention_mask = torch.cat((
            useful_past_kv_attn_mask,
            padded_kv_attn_mask
        ),
            dim=1
        )
    return attention_mask

def llm_create_1d_attn_mask(attn_mask_past_kv, attn_mask_input, cache_index=None):
    '''
    This function concatenates the attention mask corresponding to the input ids and the past kv together
    params:
    attn_mask_past_kv: the attention mask corresponding to the past kv
    attn_mask_input: the attention mask corresponding to the input (max_input tokens that the model takes)
    cache_index: cache_index determines where should the attn_mask_input be placed. If None, the input_attention mask
    is placed towards the end (assuming concat in the kv update within attention) else it is placed right after the valid kv mask.
    '''
    if cache_index is None:
        attention_mask = torch.cat((attn_mask_past_kv,
                                        attn_mask_input
                                        ),
                                        dim=1
                                        )
    else:
        attention_mask_post_valid_kv = attn_mask_past_kv[:, cache_index:]
        attention_mask_valid_kv = attn_mask_past_kv[:, :cache_index]
        attention_mask = torch.cat((
            attention_mask_valid_kv,
            attn_mask_input,
            attention_mask_post_valid_kv
        ),
            dim=1)


    return attention_mask

def llm_pad_past_kv(dummy_past_kv, unpadded_past_kv, num_hidden_layers, key_concat_axis, value_concat_axis=2, pad_to_left=True):
    """
    This function is responsible taking in current past kv and pad it using dummy kv to meet the static shape
    requirements for past kv.
    We compute the padding kv length as (Context Length - AR length) - (valid kv length).
    The shape after we pad past kv is (Context Length - AR length)

    params:
    dummy_past_kv: this corresponds to the dummy kv for one hidden layer, it is same for all the layers.
    unpadded_past_kv: this is the useful accumulated past kv (require this to obtain the length of useful past kv)
    num_hidden_layers: The number of decoder blocks in the model
    key_concat_axis: the axis to which we want to append the keys
    value_concat_axis: the axis to which we want to append the values
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """
    useful_past_kv_length = unpadded_past_kv[0][1].shape[-2] if unpadded_past_kv else 0

    # trimmed dummy kv is the final length dummy kv that will be concatenated to the unpadded_past_kv either to the left or to the right.
    trimmed_dummy_kv = (_shift(dummy_past_kv[0], key_concat_axis, useful_past_kv_length), _shift(dummy_past_kv[1], value_concat_axis, useful_past_kv_length))
    if unpadded_past_kv:
        if pad_to_left:
            padded_key_values = tuple((_concat(trimmed_dummy_kv[0], unpadded_past_kv[i][0], key_concat_axis),
                                       _concat(trimmed_dummy_kv[1], unpadded_past_kv[i][1], value_concat_axis)) for i in range(num_hidden_layers))
        else:
            padded_key_values = tuple((_concat(unpadded_past_kv[i][0], trimmed_dummy_kv[0], key_concat_axis),
                                       _concat(unpadded_past_kv[i][1], trimmed_dummy_kv[1], value_concat_axis)) for i in
                                      range(num_hidden_layers))
        return padded_key_values
    return tuple(trimmed_dummy_kv for _ in range(num_hidden_layers))

def llm_get_dummy_kv(batch_size,num_key_value_heads, head_dim, key_concat_axis, device, cache_len = None, model_context_len=None, max_input_tokens=None ):
    """
    This function determines the shape of the dummy kv using the required arguments which reflect model config
    Returns the dummy kv of fixed size each time (for a single layer). This will be used for padding the passed past kv

    params:
    batch_size: the batch size needed to create dummy kv
    model_context_len : model_context_len: maximum number of tokens that the model can consume in total
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    num_key_value_heads: the number of key value heads
    head_dim: dimension at each head
    key_concat_axis: the axis to which we want to append the keys
    device: the device to place dummy kv on, this is inferred from the unpadded_past_kv tensor if it is not None
    """

    def _cache(shape):
        return torch.zeros(shape, device=device)

    if cache_len is None:
        cache_len = model_context_len-max_input_tokens

    value = (batch_size, num_key_value_heads,cache_len , head_dim)
    key = (value[0], value[1], value[3], value[2]) if key_concat_axis == 3 else tuple(value)
    return (_cache(key), _cache(value))

def _llm_trim_padded_tensor(tensor, input_length, pad_axis=1, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful values from the returned tensor (e.g., logits or hidden states)
    since our prepared model returns fixed length tensor
    params:
        tensor: current tensor returned from the model (e.g., logits or hidden states)
        input_length: length of the valid portion of tensor
        pad_axis: dimension index of padding
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    # left padding so we remove the logits from the left & return the valid input length from the end
    if pad_to_left:
        trimmed_tensor = torch.narrow(tensor, pad_axis, tensor.shape[pad_axis] - input_length, input_length)
    # right padding, so we extract the valid input_length from the beginning
    else:
        trimmed_tensor = torch.narrow(tensor, pad_axis, 0, input_length)
    return trimmed_tensor

def llm_trim_pad_logits(cur_logits, input_ids_slice=None, inputs_embeds_slice=None, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful logits from the returned logits since our prepared model returns fixed length logits
    params:
        cur_logits: current logits returned from the model
        input_ids_slice: current input ids slice which is not padded
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    input_length = input.shape[1]

    return _llm_trim_padded_tensor(cur_logits, input_length=input_length, pad_to_left=pad_to_left)

def llm_trim_padded_hidden_states(hidden_states, input_ids_slice=None, inputs_embeds_slice=None, pad_to_left=True):
    """
    This function is responsible for stripping the non-useful hidden states from the returned hidden states
    since our prepared model returns fixed length hidden states
    params:
        hidden_states: current hidden states returned from the model
        input_ids_slice: current input ids slice which is not padded
        pad_to_left: boolean value indicating whether padding is done towards the left or right.
    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    input_length = input.shape[1]

    return _llm_trim_padded_tensor(hidden_states, input_length=input_length, pad_to_left=pad_to_left)

def llm_get_position_ids_from_attention_mask(attention_mask, max_input_tokens, model_context_len, cache_index=None):
    """
    This function computes the position ids for the tokens being fed into the model from the 1d_attn_mask.

    params:
    attention_mask: takes in the prepared attention mask needed to deduce the position ids
    max_input_tokens: the maximum tokens that can be sent to the model, in our context represents the AR length
    model_context_len : the maximum context length that can be sent to the model (this is the HF maximum length of the context)
    cache_index: the index for the starting position of kvcaches
    """

    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.clip(0, model_context_len - 1)
    if cache_index is None:
        position_ids = position_ids[..., -max_input_tokens:]
    else:
        position_ids = position_ids[..., cache_index:cache_index+max_input_tokens]
    return position_ids

def llm_pad_position_ids(position_ids_slice, max_input_tokens, pad_value=0, pad_to_left = True):
    """
    This function pads the position_ids since slice may return position_ids that is smaller than what the model accepts (AR len)

    params:
    position_ids_slice: the current position_ids slice that is passed into the model in the current invocation
    max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    pad_value: padding value, this is defaulted to 0
    pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """

    assert position_ids_slice is not None
    assert position_ids_slice.dim() == 2

    batch_size, pos_ids_len = position_ids_slice.shape

    if pos_ids_len < max_input_tokens:
        pad_pos_ids = torch.full((batch_size, max_input_tokens-pos_ids_len), pad_value,
                                 dtype=position_ids_slice.dtype, device=position_ids_slice.device)

        if pad_to_left:
            position_ids = torch.cat((pad_pos_ids, position_ids_slice), dim=-1)
        else:
            position_ids = torch.cat((position_ids_slice, pad_pos_ids), dim=-1)

        return position_ids
    else:
        return position_ids_slice
