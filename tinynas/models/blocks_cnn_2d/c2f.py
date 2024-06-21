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

    def get_deepmad_forward(self, **kwarg):
        #alpha_config = {'alpha1':1, 'alpha2':1}
        return [
            # first
            np.log(np.sqrt(self.in_channels ** kwarg["alpha1"])) +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) +
            
            # second
            np.log(np.sqrt(self.c ** kwarg["alpha1"])) +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) 
        ]

    def get_width(self):
        width = float(self.in_channels * self.kernel_size ** 2 * self.c * self.kernel_size ** 2 // self.groups)
        return width



class C2f(nn.Module):

    def __init__(self,
                 structure_info,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'C2f',
            'in': in_channels,
            'out': out_channels,
            'k': kernel_size,
            'n': number (default=1),
            's': stride (default=1),
            'g': grouping (default=1),
            'e': expansion (default=0.5),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''

        super().__init__()

        #if 'class' in structure_info:
        #    assert structure_info['class'] == self.__class__.__name__

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
            'out': 2 * self.c,
            'k': 1,
            'act': self.act
        }

        conv2_info = {
            'in': (2 + self.n) * self.c,
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

        self.m = nn.ModuleList(Bottleneck(btn_info, **kwargs) for _ in range(self.n))

        # self.block_list = []
        # self.block_list.append(self.conv1)
        # self.block_list.append(self.conv2)
        # self.block_list.append(self.m)

        self.model_size = 0.0
        self.flops = 0.0
        self.model_size += self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops += self.conv1.get_flops(1.0) + self.conv2.get_flops(1.0)

        for block in self.m:
            self.model_size += block.get_model_size()
            self.flops += block.get_flops(1.0)


    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


    def get_model_size(self, return_list=False):
        if return_list:
            return [self.model_size]
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        layers = 2
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

    def get_deepmad_forward(self, **kwarg):
        
        #alpha_config = {'alpha1':1, 'alpha2':1}
        return [
            # conv1
            np.log(np.sqrt(self.in_channels ** kwarg["alpha1"])) +
            np.log(np.sqrt(1 ** (2 * kwarg["alpha2"]))) +
            
            # conv2
            np.log(np.sqrt(((2 + self.n) * self.c) ** kwarg["alpha1"])) +
            np.log(np.sqrt(1 ** (2 * kwarg["alpha2"]))) +
            
            # m
            np.log(np.sqrt(self.c ** kwarg["alpha1"])) * self.n +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) * self.n
        ]

    def get_bottleneck_entropy(self,in_resolution,**kwarg):
        
        out_resolution=in_resolution
        conv1=(np.log(np.sqrt(self.in_channels ** kwarg["alpha1"])) +
              np.log(np.sqrt(1 ** (2 * kwarg["alpha2"]))) +
              np.log(np.sqrt(out_resolution ** (2 * kwarg["alpha2"]))) +
              np.log(np.sqrt((self.c*2 ) ** kwarg["alpha1"]))
              )
        
        
        conv2=(np.log(np.sqrt(self.c ** kwarg["alpha1"]))  +
               np.log(np.sqrt( (self.n + 2) ** kwarg["alpha1"]))  +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(out_resolution ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(self.out_channels ** kwarg["alpha1"]))
            )
        
        
        btn=(np.log(np.sqrt(self.c ** kwarg["alpha1"])) +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(out_resolution ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(self.c ** kwarg["alpha1"]))
            )

        return min(conv1,conv2,btn)


    def get_width(self):
        width = float(self.in_channels * 1 ** 2 
                      * (2 + self.n) * self.c * 1 ** 2)
        for block in self.m:
            width *= block.get_width()
        return [width]



__module_blocks__ = {
    'C2f': C2f
}


