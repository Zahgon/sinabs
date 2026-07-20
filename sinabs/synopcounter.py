import warnings

import torch
import torch.nn as nn

import sinabs.layers as sl
from sinabs.layers import NeuromorphicReLU


def spiking_hook(self, input_, output):
    pass


def synops_hook(self, input_, output):
    pass


class SNNAnalyzer:

    def __init__(self, model: torch.nn.Module, dt: float = 1.0):
        warnings.warn(
            "SNNAnalyzer is deprecated. Please use `sinabs.hooks` instead",
            DeprecationWarning,
        )
        self.model = model
        self.dt = dt
        self.handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        pass

    def __del__(self):
        for handle in self.handles:
            handle.remove()

    def get_layer_statistics(self, average: bool = False) -> dict:
        pass

    def get_model_statistics(self, average: bool = False) -> dict:
        pass

    def reset(self):
        pass


class SynOpCounter:

    def __init__(self, modules, sum_activations=True):
        warnings.warn(
            "SNNAnalyzer is deprecated. Please use `sinabs.hooks` instead",
            DeprecationWarning,
        )
        self.modules = []
        for module in modules:
            if isinstance(module, NeuromorphicReLU) and module.fanout > 0:
                self.modules.append(module)

        if len(self.modules) == 0:
            raise ValueError("No NeuromorphicReLU found in module list.")

        self.sum_activations = sum_activations

    def __call__(self):
        synops = []
        for module in self.modules:
            synops.append(module.activity)

        if self.sum_activations:
            synops = torch.stack(synops).sum()
        return synops
