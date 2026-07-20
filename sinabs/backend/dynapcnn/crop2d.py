from typing import List, Tuple, Union

import numpy as np
from torch import nn

ArrayLike = Union[np.ndarray, List, Tuple]


class Crop2d(nn.Module):

    def __init__(
        self,
        cropping: ArrayLike = ((0, 0), (0, 0)),
    ):
        """Crop input to the the rectangle dimensions.

        :param cropping: ((top, bottom), (left, right))
        """
        super().__init__()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        pass

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        pass

    def __repr__(self):
        return f"Crop2d(({self.top_crop}, {self.bottom_crop}), ({self.left_crop}, {self.right_crop}))"
