import torch

from sinabs.activation import Quantize, StochasticRounding


class NeuromorphicReLU(torch.nn.Module):

    def __init__(self, quantize=True, fanout=1, stochastic_rounding=False):
        super().__init__()
        self.quantize = quantize
        self.stochastic_rounding = stochastic_rounding
        self.fanout = fanout

    def forward(self, inp):
        pass
