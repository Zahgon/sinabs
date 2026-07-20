from typing import Tuple

import torch
import torch.nn as nn


class FlipDims(nn.Module):
    def __init__(
        self, flip_x: bool = False, flip_y: bool = False, swap_xy: bool = False
    ):
        super().__init__()
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy

    def forward(self, data):
        pass

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        pass
