import argparse
import logging
import time
from abc import ABCMeta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from tinynas.searchers import build_searcher
from tinynas.utils.dict_action import DictAction
from tinynas.strategy import Strategy


class VisualizeErfScore(metaclass=ABCMeta):
    def __init__(self, in_channels=12, image_size=320, weights=[0, 1, 1, 1], threshold=0.65, logger=None, **kwargs):
        self.resolution = image_size
        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold

        self.logger = logger or logging
        self.dist_mat = self.get_dist_mat(self.resolution, self.resolution).to(self.device)
        self.samples = torch.randn(1, in_channels, self.resolution, self.resolution, requires_grad=True).to(self.device)

    def __call__(self, model):
        info = {}
        timer_start = time.time()
        model = model.to(self.device)
        # set optimizer
        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
        optimizer.zero_grad()
        avg_nas_score = self.get_model_erf(model, self.dist_mat, optimizer)
        timer_end = time.time()
        info['avg_nas_score'] = avg_nas_score
        info['time'] = timer_end - timer_start
        self.logger.debug('avg_score:%s, consume time is %f ms' %
                          (avg_nas_score, info['time'] * 1000))

        return info

    def get_input_grad(self, model, optimizer):
        outputs = model(self.samples)
        grad_maps = []
        inp = self.samples
        base_wh = self.samples.shape[2:]
        for out in outputs:
            print(out.size())

        for out in outputs:
            optimizer.zero_grad()
            out_size = out.size()
            central_point = torch.nn.functional.relu(out[:, :, out_size[2] // 2, out_size[3] // 2]).mean()
            grad = torch.autograd.grad(central_point, inp, retain_graph=False)
            inp = out
            grad = grad[0]
            grad = torch.nn.functional.relu(grad)
            grad = torch.nn.functional.interpolate(grad, base_wh, mode='nearest')
            aggregated = grad.mean((0, 1))

            grad_maps.append(aggregated)
        return grad_maps

    def get_model_erf(self, model, dist_mat, optimizer):
        contribution_scores = self.get_input_grad(model, optimizer)
        value = 0
        for idx, grad_map in enumerate(contribution_scores):
            grad_map = torch.log(grad_map + 1)
            grad_map = grad_map / grad_map.max()
            grad_map[grad_map < self.threshold] = 0
            grad_map[grad_map >= self.threshold] = 1
            print(idx, (dist_mat * grad_map).max().item())

            import matplotlib.pyplot as plt
            plt.imshow(grad_map.cpu())
            plt.colorbar()
            plt.savefig('{}.png'.format(idx, idx))
            plt.clf()

            value += grad_map * self.weights[idx]

        return value.cpu()

    def get_dist_mat(self, hsize, wsize):
        dist_mat = torch.zeros((hsize, wsize))
        for i in range(hsize):
            for j in range(wsize):
                dist_mat[i, j] = (hsize // 2 - i) ** 2 + (wsize // 2 - j) ** 2
        return dist_mat.sqrt()


def parse_args():
    parser = argparse.ArgumentParser(description='Search a network model')
    parser.add_argument('config', help='search config file path')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    kwargs = dict(cfg_file=args.config)
    if args.cfg_options is not None:
        kwargs['cfg_options'] = args.cfg_options
    searcher = build_searcher(default_args = kwargs)
    model = searcher.strategy.super_model.build(searcher.init_structure_info)

    image_size = searcher.cfg.image_size
    in_channels = searcher.init_structure_info[0]['in']
    visualizeErfScore=VisualizeErfScore(in_channels=in_channels, image_size=image_size, threshold=searcher.cfg.score.threshold)
    visualizeErfScore(model)


if __name__ == '__main__':
    main()
