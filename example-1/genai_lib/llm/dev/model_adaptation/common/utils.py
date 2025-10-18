#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides utilities that are common across models. These utilities are needed because of model adaptations """

from typing import Union, Tuple
import functools
import torch

from genai_lib.llm.utils import _concat, _shift

KEY_CONCAT_AXIS = 3
VALUE_CONCAT_AXIS = 2

def llm_update_kv_cache(unpadded_past_kv, current_key_values,  key_concat_axis=KEY_CONCAT_AXIS, value_concat_axis=VALUE_CONCAT_AXIS, input_ids_slice = None, inputs_embeds_slice=None, pad_to_left=True):
    """
    This function concats the KV cache that the model outputs in the current iteration (unpadded_past_kv) with the KV$ that the model has accumulated so far(unpadded_past_kv)
    1. remove the non-useful padding kv from the current_key_values depending on whether it was padded to left or to right
    2. concatenate the stripped current kv with past useful kv if it exists

    params:
    1. unpadded_past_kv: the unpadded useful kv that is accumulated from the previous model invokations
    2. current_key_values: current padded kv returned from the model
    3. key_concat_axis: the axis to which we want to append the keys
    4. value_concat_axis: the axis to which we want to append the values
    5. input_ids_slice: the slice of inputs returned from the iterator (this is before any padding has been applied to meet the static shape requirement)
    6. inputs_embeds_slice: the slice of inputs returned from the iterator (this is before any padding has been applied to meet the static shape requirement)
    7. pad_to_left: boolean value indicating whether padding is done towards the left or right.

    """
    input = input_ids_slice if input_ids_slice is not None else inputs_embeds_slice

# TODO determine whether we need to trim or not based on the current_pad_len, if negative do not trim. min set to 0
    trimmed_current_key_values = trim_current_kv(current_key_values, input , key_concat_axis, value_concat_axis, pad_to_left)
    # slicing in place before sending to concat function to avoid  memory spiking.
    if unpadded_past_kv:
        concatenated_key_values = tuple((_concat(unpadded_key, current_key, key_concat_axis),
                                     _concat(unpadded_value, current_value, value_concat_axis))
                                    for (unpadded_key, unpadded_value), (current_key, current_value) in zip(unpadded_past_kv, trimmed_current_key_values))
        return concatenated_key_values

    return trimmed_current_key_values

def trim_current_kv(current_key_values, input,  key_concat_axis, value_concat_axis=2, pad_to_left=True):
    '''
    params:
    1. current_key_values: current padded/ unpadded kv returned from the model
    if the current_pad_length we compute is positive, means the current keys have padding which need to be removed. But if the current_pad_length is negative, we do not remove anything since the user has already trimmed the kv before. (This is possible when this API gets invoked from the update_kv_cache API where the current_kv only refers to the selected draft and valid token, this will be smaller than the input_ids_slice size.
    '''
    input_length = input.shape[1]
    # limit the value of this to be non-negative, if it is negative, we assign it to be 0, hence no shifting.
    current_pad_length = max(0, (current_key_values[0][1].shape[2] - input_length))
    return tuple((_shift(current_key, key_concat_axis, current_pad_length, pad_to_left), _shift(current_value, value_concat_axis, current_pad_length, pad_to_left)) for current_key, current_value in current_key_values)


def llm_extract_past_kv_at_idx(accepted_token_index_list, current_kv, is_key_transposed=True):
    '''
    This function is responsible for extracting the kv cache at indices in the accepted tokens index list.
    params:
    1. accepted_token_index_list: the list containing indices of the accepted tokens
    2. current_kv: current KV from which the subset KV is extracted.
    3. is_key_transposed: true if key cache is transposed
    '''
    n_hidden_layers = len(current_kv)
    pruned_keys = [[] for _ in range(n_hidden_layers)]
    pruned_values = [[] for _ in range(n_hidden_layers)]

    for n_layer in range(n_hidden_layers):
        if is_key_transposed:
            pruned_keys[n_layer] =  current_kv[n_layer][0][:, :, :, accepted_token_index_list]
        else:
            pruned_keys[n_layer] =  current_kv[n_layer][0][:, :, accepted_token_index_list, :]
        pruned_values[n_layer] = current_kv[n_layer][1][:, :, accepted_token_index_list, :]

    return tuple((pruned_keys[i], pruned_values[i]) for i in range(n_hidden_layers))

