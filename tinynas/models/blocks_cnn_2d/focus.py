# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks_basic import ConvKXBNRELU



class Focus(nn.Module):
    """Focus width and height information into channel space."""
    '''
    :param structure_info: {
        'class': 'Focus',
        'in': in_channels,
        'out': out_channels,
        'k': kernel_size,
        's': stride (default=1),
        'act': activation (default=relu),
    }
    :param NAS_mode:
    '''

    def __init__(self,
                 structure_info,
                 **kwargs):
        super().__init__()
        self.in_channels = structure_info['in']*4
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']

        if 'act' in structure_info:
            self.act = structure_info['act']
        else:
            self.act = 'relu'

        conv_info = {
            'in': self.in_channels,
            'out': self.out_channels,
            'k': self.kernel_size,
            'act': self.act
        }

        self.conv = ConvKXBNRELU(conv_info, **kwargs)
        
        self.model_size = 0.0
        self.flops = 0.0
        self.model_size += self.conv.get_model_size()
        self.flops = self.flops + self.conv.get_flops(1.0)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    def get_model_size(self, return_list=False):
        if return_list:
            return [self.model_size]
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        return 1

    def get_output_resolution(self, input_resolution):
        return input_resolution // 2

    def get_params_for_trt(self, input_resolution):
        raise NotImplementedError

    def get_num_channels_list(self):
        raise NotImplementedError


    def get_madnas_forward(self, **kwarg):
        raise NotImplementedError

    def get_max_feature_num(self, resolution):
        raise NotImplementedError

    def get_width(self):
        width = float(self.in_channels * self.kernel_size ** 2)
        return [width]

    def get_block_num(self):
        return 1

__module_blocks__ = {
    'Focus': Focus
}


