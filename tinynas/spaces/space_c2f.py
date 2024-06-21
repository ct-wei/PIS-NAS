from .base import CnnBaseSpace
from .builder import SPACES
from .mutator import build_mutator

@SPACES.register_module(module_name = 'space_c2f')
class SpaceC2f(CnnBaseSpace):
    def __init__(self, name = None,
                    image_size = 224,
                    block_num = 2,
                    exclude_stem = False,
                    maximum_channel =2048,
                    channel_range_list = None,
                    kernel_size_list = None,
                    **kwargs):

        super().__init__(name, image_size = image_size,
                        block_num = block_num,
                        exclude_stem = exclude_stem,
                        **kwargs)
        defualt_channel_range_list = [None, None, None, None, None, None, None, None, None]
        defualt_kernel_size_list = [3]

        channel_range_list = channel_range_list or defualt_channel_range_list
        kernel_size_list = kernel_size_list or defualt_kernel_size_list

        kwargs = dict(stem_mutate_method_list = ['out'] ,
            mutate_method_list = ['out', 'n'],
            the_maximum_channel = maximum_channel,
            channel_range = channel_range_list,
            search_kernel_size_list = kernel_size_list,
            search_layer_list = [-2, -1, 1, 2],
            search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5],
        )
        for n in ['ConvKXBNRELU', 'C2f']:
            self.mutators[n] = build_mutator(n, kwargs)

