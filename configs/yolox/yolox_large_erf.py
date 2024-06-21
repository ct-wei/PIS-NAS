# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/yolovx_erf_ef6e-2/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 640

maximum_channel = 1024

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [
        {'class': 'Focus', 'act': 'silu', 'in': 3, 'out': 64, 's': 2, 'k': 3}, # Focus
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 64, 'out': 128, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 128, 'out': 128,'s': 1,'n': 3, 'k': 1}, # 2
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 128, 'out': 256, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 256, 'out': 256,'s': 1,'n': 9, 'k': 1}, # 4
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 256, 'out': 512, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 512, 'out': 512,'s': 1,'n': 9, 'k': 1}, # 6
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 512, 'out': 1024, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 1024, 'out': 1024,'s': 1,'n': 3, 'k': 1}, # 8
    ]
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 29e8),
    dict(type = "efficient_score", budget=0.4),
    dict(type = "model_size", budget=15e5),
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
    type='space_csp',
    image_size=image_size,
    maximum_channel=maximum_channel,
    channel_range_list=[None, None, [48, 96], None, [96, 192],  None, [192, 384], None, [384, 768]],
    kernel_size_list=[3])

""" Search config """
search=dict(
    minor_mutation=False,  # whether fix the stage layer
    minor_iter=100000,  # which iteration to enable minor_mutation
    popu_size=512,
    num_random_nets=500000,  # the searching iterations
    sync_size_ratio=1.0,  # control each thread sync number: ratio * popu_size
    num_network=1,
)
