from typing import Optional, Tuple

import torch.nn as nn

from sinabs.layers import SumPool2d
from sinabs.utils import expand_to_pair

from .crop2d import Crop2d
from .flipdims import FlipDims


class DVSLayer(nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int],
        pool: Tuple[int, int] = (1, 1),
        crop: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        merge_polarities: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        swap_xy: bool = False,
        disable_pixel_array: bool = True,
    ):
        super().__init__()

        self.merge_polarities = merge_polarities
        self.disable_pixel_array = disable_pixel_array

        if len(input_shape) != 2:
            raise ValueError(
                f"Input shape should be 2 dimensional but input_shape={input_shape} was given."
            )
        if merge_polarities:
            self.input_shape: Tuple[int, int, int] = (1, *input_shape)
        else:
            self.input_shape: Tuple[int, int, int] = (2, *input_shape)

        self.pool_layer = SumPool2d(pool)

        if crop is None:
            num_channels, height, width = self.get_output_shape_after_pooling()
            crop = ((0, height), (0, width))
        self.crop_layer = Crop2d(crop)

        self.flip_layer = FlipDims(flip_x, flip_y, swap_xy)

    @classmethod
    def from_layers(
        cls,
        input_shape: Tuple[int, int, int],
        pool_layer: Optional[SumPool2d] = None,
        crop_layer: Optional[Crop2d] = None,
        flip_layer: Optional[FlipDims] = None,
        disable_pixel_array: bool = True,
    ) -> "DVSLayer":
        pass

    @property
    def input_shape_dict(self) -> dict:
        pass

    def get_output_shape_after_pooling(self) -> Tuple[int, int, int]:
        """Get the shape of data just after the pooling layer.

        Returns:
            (channel, height, width)
        """
        channel_count, input_size_y, input_size_x = self.input_shape

        if self.merge_polarities:
            channel_count = 1

        pooling = self.get_pooling()
        output_size_x = input_size_x // pooling[1]
        output_size_y = input_size_y // pooling[0]
        return channel_count, output_size_y, output_size_x

    def get_output_shape_dict(self) -> dict:
        pass

    def get_config_dict(self) -> dict:
        crop = self.get_roi()
        cut = {"x": crop[1][1] - 1, "y": crop[0][1] - 1}
        origin = {"x": crop[1][0], "y": crop[0][0]}
        pooling = {"y": self.get_pooling()[0], "x": self.get_pooling()[1]}

        return {
            "merge": self.merge_polarities,
            "mirror": self.get_flip_dict(),
            "mirror_diagonal": self.get_swap_xy(),
            "cut": cut,
            "origin": origin,
            "pooling": pooling,
            "pass_sensor_events": not self.disable_pixel_array,
        }

    def forward(self, data):
        pass

    def get_pooling(self) -> Tuple[int, int]:
        """Pooling kernel shape.

        Returns:
            (ky, kx)
        """
        return expand_to_pair(self.pool_layer.kernel_size)

    def get_roi(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """The coordinates for ROI. Note that this is not the same as crop parameter passed during
        the object construction.

        Returns:
            ((top, bottom), (left, right))
        """
        _, h, w = self.get_output_shape_after_pooling()
        return (
            (self.crop_layer.top_crop, self.crop_layer.bottom_crop),
            (self.crop_layer.left_crop, self.crop_layer.right_crop),
        )

    def get_output_shape(self) -> Tuple[int, int, int]:
        pass

    def get_flip_dict(self) -> dict:
        """Configuration dictionary for x, y flip.

        Returns:
            dict
        """

        return {"x": self.flip_layer.flip_x, "y": self.flip_layer.flip_y}

    def get_swap_xy(self) -> bool:
        """True if XY has to be swapped.

        Returns:
            bool
        """
        return self.flip_layer.swap_xy
