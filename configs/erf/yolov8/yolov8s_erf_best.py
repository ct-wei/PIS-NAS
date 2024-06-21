# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/yolov8s_erf_be/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 512

maximum_channel = 1024

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [ {'act': 'silu', 'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'out': 32, 's': 2},
                         {'act': 'silu', 'class': 'ConvKXBNRELU', 'in': 32, 'k': 3, 'out': 64, 's': 2},
                         {'act': 'silu', 'class': 'C2f', 'in': 64, 'k': 3, 'n': 1, 'out': 64, 's': 1},
                         {'act': 'silu', 'class': 'ConvKXBNRELU', 'in': 64, 'k': 3, 'out': 96, 's': 2},
                         {'act': 'silu', 'class': 'C2f', 'in': 96, 'k': 3, 'n': 3, 'out': 112, 's': 1},
                         {'act': 'silu', 'class': 'ConvKXBNRELU', 'in': 112, 'k': 3, 'out': 192, 's': 2},
                         {'act': 'silu', 'class': 'C2f', 'in': 192, 'k': 3, 'n': 2, 'out': 256, 's': 1},
                         {'act': 'silu', 'class': 'ConvKXBNRELU', 'in': 256, 'k': 3, 'out': 384, 's': 2},
                         {'act': 'silu', 'class': 'C2f', 'in': 384, 'k': 3, 'n': 2, 'out': 464, 's': 1}]
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 40e8),
    dict(type = "model_size", budget= 4.8e6),
    dict(type = "bottleneck_entropy", budget= -8.8),
]

""" Score config """
score = dict(
    type='erfnas',
    depth_scales=[1, 2, 2, 1],
    depth_penalty_ratio=8.,
    in_channels=model['structure_info'][0]['in'],
    image_size=image_size,
    weights=[0, 1, 1, 1],
    threshold=0.1,

    
)

""" Space config """
space = dict(
    type='space_c2f',
    image_size=image_size,
    maximum_channel=maximum_channel,
    kernel_size_list=[3])

""" Search config """
search=dict(
    minor_mutation=False,  # whether fix the stage layer
    minor_iter=100000,  # which iteration to enable minor_mutation
    popu_size=512,
    num_random_nets=200000,  # the searching iterations
    sync_size_ratio=1.0,  # control each thread sync number: ratio * popu_size
    num_network=1,
)
