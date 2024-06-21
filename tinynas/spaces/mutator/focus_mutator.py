# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import bisect
import copy
import os
import random
import sys

from ..space_utils import smart_round
from .builder import *


@MUTATORS.register_module(module_name = 'Focus')
class focusMutator():
    def __init__(self,
                mutate_method_list,
                channel_range,
                search_channel_list,
                search_kernel_size_list,
                search_layer_list,
                the_maximum_channel,
                *args,
                **kwargs):
        self.channel_range = channel_range

        minor_mutation_list = ['out']
        kwargs.update(dict(candidates = minor_mutation_list))
        self.minor_method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = mutate_method_list))
        self.method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = search_channel_list), the_maximum_channel= the_maximum_channel)
        self.channel_mutator = build_channel_mutator(kwargs)

        kwargs = dict(candidates = search_kernel_size_list )
        self.kernel_mutator = build_kernel_mutator(kwargs)

        kwargs = dict(candidates = search_layer_list)
        self.layer_mutator = build_layer_mutator(kwargs)

    def __call__(self, block_id, structure_info_list, minor_mutation = False, *args, **kwargs):

        structure_info = structure_info_list[block_id]
        if block_id < len(structure_info_list) - 1:
            structure_info_next = structure_info_list[block_id + 1]
        structure_info = copy.deepcopy(structure_info)

        # coarse2fine mutation flag, only mutate the channels' output
        random_mutate_method = self.minor_method_mutator() if minor_mutation else self.method_mutator()


        if random_mutate_method == 'out':
            old_v = structure_info['out']
            new_out = self.channel_mutator(structure_info['out'])

            # Add the contraint: output_channel should be in range [min, max]
            if self.channel_range[block_id] is not None:
                this_min, this_max = self.channel_range[block_id]
                new_out = max(this_min, min(this_max, new_out))

            # Add the constraint: output_channel > input_channel
            new_out = max(structure_info['in'], new_out)
            if block_id < len(structure_info_list) - 1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out

        if random_mutate_method == 'k':
            new_k = self.kernel_mutator(structure_info['k'])
            structure_info['k'] = new_k

        return structure_info
