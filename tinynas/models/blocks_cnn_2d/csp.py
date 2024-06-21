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

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        structure_info,
        **kwargs
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        
        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']

        if 'act' in structure_info:
            self.act = structure_info['act']
        else:
            self.act = 'relu'

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'e' in structure_info:
            self.e = structure_info['e']
        else:
            self.e = 0.5

        if 'n' in structure_info:
            self.n = int(structure_info['n'])
        else:
            self.n = 1

        self.c = int(self.out_channels * self.e)


        conv1_info = {
            'in': self.in_channels,
            'out': self.c,
            'k': 1,
            'act': self.act
        }
            
        conv2_info = {
            'in': self.in_channels,
            'out': self.c,
            'k': 1,
            'act': self.act
        }

        conv3_info = {
            'in': self.c*2,
            'out': self.out_channels,
            'k': 1,
            'act': self.act
        }
        
        btn_info = {
            'in': self.c,
            'out': self.c,
            'k': self.kernel_size,
            'g': self.groups,
            'e': 1,
            'act': self.act
        }
        
        self.conv1 = ConvKXBNRELU(conv1_info, **kwargs)
        self.conv2 = ConvKXBNRELU(conv2_info, **kwargs)
        self.conv3 = ConvKXBNRELU(conv3_info, **kwargs)
        self.m = nn.ModuleList(Bottleneck(btn_info, **kwargs) for _ in range(self.n))


        self.model_size = 0.0
        self.flops = 0.0
        self.model_size += self.conv1.get_model_size() + self.conv2.get_model_size()+self.conv3.get_model_size()
        self.flops += self.conv1.get_flops(1.0) + self.conv2.get_flops(1.0) + self.conv3.get_flops(1.0)

        for block in self.m:
            self.model_size += block.get_model_size()
            self.flops += block.get_flops(1.0)


    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        for module in self.m:
            x_1 = module(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)
    
    def get_model_size(self, return_list=False):
        if return_list:
            return [self.model_size]
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        layers = 3
        for block in self.m:
            layers += block.get_layers()
        return layers
    
    def get_block_num(self):
        return 1


    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_params_for_trt(self, input_resolution):
        raise NotImplementedError

    def get_num_channels_list(self):
        raise NotImplementedError

    def get_madnas_forward(self, **kwarg):
        raise NotImplementedError

    def get_max_feature_num(self, resolution):
        raise NotImplementedError

    def get_width(self):
        width = float(self.in_channels * 1 ** 2 
                      * self.c*2 * 1 **2
                      * self.c*2 * 1 **2)
        for block in self.m:
            width *= block.get_width()
        return [width]


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self,
                 structure_info,
                 **kwargs):
        super().__init__()

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.groups = structure_info['g']
        self.e = structure_info['e']
        self.act = structure_info['act']

        self.c = int(self.out_channels * self.e)  # hidden channels
        conv1_info = {
            'in': self.in_channels,
            'out': self.c,
            'k': self.kernel_size,
            'act': self.act
        }

        conv2_info = {
            'in': self.c,
            'out': self.out_channels,
            'k': self.kernel_size,
            'g': self.groups,
            'act': self.act
        }
        self.conv1 = ConvKXBNRELU(conv1_info, **kwargs)
        self.conv2 = ConvKXBNRELU(conv2_info, **kwargs)
        self.add = self.in_channels == self.out_channels

        self.model_size = 0.0
        self.flops = 0.0
        self.model_size += self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(1.0) + self.conv2.get_flops(1.0)


    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

    def get_model_size(self, return_list=False):
        if return_list:
            return [self.model_size]
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        return 2

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_params_for_trt(self, input_resolution):
        raise NotImplementedError

    def get_num_channels_list(self):
        raise NotImplementedError

    def get_madnas_forward(self, **kwarg):
        raise NotImplementedError

    def get_max_feature_num(self, resolution):
        raise NotImplementedError

    def get_width(self):
        width = float(self.in_channels * self.kernel_size ** 2 * self.c * self.kernel_size ** 2 // self.groups)
        return width





__module_blocks__ = {
    'csp': CSPLayer
}


