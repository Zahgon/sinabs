from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from sinabs.cnnutils import conv_output_size

ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingMaxPooling2dLayer(nn.Module):

    def __init__(
        self,
        pool_size: ArrayLike,
        strides: Optional[ArrayLike] = None,
        padding: ArrayLike = (0, 0, 0, 0),
    ):
        super().__init__()
        self.padding = padding
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        if padding == (0, 0, 0, 0):
            self.pad = None
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)

        self.spikes_number = None

    def forward(self, binary_input):
        pass

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        pass


class SumPool2d(torch.nn.LPPool2d):

    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super().__init__(1, kernel_size, stride, ceil_mode)
