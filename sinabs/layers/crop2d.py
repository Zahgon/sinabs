from typing import List, Tuple, Union

import numpy as np
from torch import nn

ArrayLike = Union[np.ndarray, List, Tuple]


class Cropping2dLayer(nn.Module):

    def __init__(
        self,
        cropping: ArrayLike = ((0, 0), (0, 0)),
    ):
        super().__init__()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        pass

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        pass
