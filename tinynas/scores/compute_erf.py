import logging
import time
from abc import ABCMeta

import torch
import torch.optim as optim

from .builder import SCORES


@SCORES.register_module(module_name = 'erfnas')
class ComputeErfScore(metaclass=ABCMeta):
    def __init__(self, batch=8, in_channels=12, image_size=320, weights=[0, 1, 1, 1], threshold=0.65, logger=None, **kwargs):
        self.resolution = image_size
        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold

        self.logger = logger or logging
        self.dist_mat = self.get_dist_mat(self.resolution, self.resolution).to(self.device)
        self.samples = torch.randn(batch, in_channels, self.resolution, self.resolution, requires_grad=True).to(self.device)

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
        if not isinstance(outputs[0], (list, tuple)):
            outputs = [outputs]
        outputs = outputs[-1]
        grad_maps = []
        inp = self.samples
        base_wh = self.samples.shape[2:]
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
            grad_map = (dist_mat * grad_map).max()
            value += grad_map * self.weights[idx]

        return value.cpu()

    def get_dist_mat(self, hsize, wsize):
        dist_mat = torch.zeros((hsize, wsize))
        for i in range(hsize):
            for j in range(wsize):
                dist_mat[i, j] = (hsize // 2 - i) ** 2 + (wsize // 2 - j) ** 2
        return dist_mat.sqrt()
