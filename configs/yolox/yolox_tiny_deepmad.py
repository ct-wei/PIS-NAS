# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/yolox_tiny_erf_ef6e-2/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 640

maximum_channel = 1024

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [
        {'class': 'Focus', 'act': 'silu', 'in': 3, 'out': 24, 's': 2, 'k': 3}, # Focus
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 24, 'out': 48, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 48, 'out': 48,'s': 1,'n': 1, 'k': 1}, # 2
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 48, 'out': 96, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 96, 'out': 96,'s': 1,'n': 2, 'k': 1}, # 4
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 96, 'out': 192, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 192, 'out': 192,'s': 1,'n': 2, 'k': 1}, # 6
        {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 192, 'out': 384, 's': 2, 'k': 3},
        {'class': 'csp', 'act': 'silu', 'in': 384, 'out': 384,'s': 1,'n': 1, 'k': 1}, # 8
    ]
)

    # tiny structure_info = [
    #     {'class': 'Focus', 'act': 'silu', 'in': 3, 'out': 24, 's': 2, 'k': 3}, # Focus
    #     {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 24, 'out': 48, 's': 2, 'k': 3},
    #     {'class': 'csp', 'act': 'silu', 'in': 48, 'out': 48,'s': 1,'n': 1, 'k': 1}, # 2
    #     {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 48, 'out': 96, 's': 2, 'k': 3},
    #     {'class': 'csp', 'act': 'silu', 'in': 96, 'out': 96,'s': 1,'n': 3, 'k': 1}, # 4
    #     {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 96, 'out': 192, 's': 2, 'k': 3},
    #     {'class': 'csp', 'act': 'silu', 'in': 192, 'out': 192,'s': 1,'n': 3, 'k': 1}, # 6
    #     {'class': 'ConvKXBNRELU', 'act': 'silu', 'in': 192, 'out': 384, 's': 2, 'k': 3},
    #     {'class': 'csp', 'act': 'silu', 'in': 384, 'out': 384,'s': 1,'n': 1, 'k': 1}, # 8
    # ]
    
""" Budget config """
budgets = [
    dict(type = "flops", budget = 3e9),
    dict(type = "efficient_score", budget=10e9),
    # dict(type = "model_size", budget=14e5),
]

""" Score config """
score = dict(
    type='deepmad',
    depth_scales=[1, 2, 2, 1],
    depth_penalty_ratio=8.,
    in_channels=model['structure_info'][0]['in'],
    image_size=image_size,
    weights=[0, 1, 1, 1],
    # threshold=0.1,
)

""" Space config """
space = dict(
    type='space_csp',
    image_size=image_size,
    maximum_channel=maximum_channel,
    channel_range_list=[None, [24, 48], None, [48, 96],  None, [96, 192], None, [192, 384], None],
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
