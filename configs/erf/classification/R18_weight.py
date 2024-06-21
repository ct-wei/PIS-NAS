# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/R18_erf_be_weight_ds/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 224

maximum_channel = 1024

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = 
    [ 
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3}, \
        {'class': 'SuperResK1KXK1', 'in': 32, 'out': 256, 's': 2, 'k': 3, 'L': 1, 'btn': 64}, \
        {'class': 'SuperResK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, \
        {'class': 'SuperResK1KXK1', 'in': 512, 'out': 768, 's': 2, 'k': 3, 'L': 1, 'btn': 256}, \
        {'class': 'SuperResK1KXK1', 'in': 768, 'out': 1024, 's': 1, 'k': 3, 'L': 1, 'btn': 256}, \
        {'class': 'SuperResK1KXK1', 'in': 1024, 'out': 2048, 's': 2, 'k': 3, 'L': 1, 'btn': 512}, \
    ]
    
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 1.82e9),
    dict(type = "model_size", budget = 11.69e6),
    dict(type = "bottleneck_entropy", budget= -8.0)
]

""" Score config """
score = dict(
    type='erfnas',
    depth_scales=[1, 1, 2, 1],
    depth_penalty_ratio=8.,
    in_channels=model['structure_info'][0]['in'],
    image_size=image_size,
    weights=[0, 0, 0, 1],
    threshold=0.1,

    
)

""" Space config """
space = dict(
    type = 'space_k1kxk1',
    image_size = image_size,
    )

""" Search config """
search=dict(
    minor_mutation=False,  # whether fix the stage layer
    minor_iter=100000,  # which iteration to enable minor_mutation
    popu_size=512,
    num_random_nets=200000,  # the searching iterations
    sync_size_ratio=1.0,  # control each thread sync number: ratio * popu_size
    num_network=1,
)
