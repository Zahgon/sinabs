import torch.nn as nn

from sinabs.activation import Quantize


class QuantizeLayer(nn.Module):

    def __init__(self, quantize=True):
        super().__init__()
        self.quantize = quantize

    def forward(self, data):
        pass
