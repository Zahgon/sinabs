from typing import Tuple
import numpy as np


class ValidateMapping:
    def __init__(
        self,
        input_feature_size: int,
        output_feature_size: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        input_dimension: Tuple[int, int] = [64, 64],
        conv_2d: bool = True,
    ):
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_dimension = input_dimension

        if not conv_2d:
            if (
                kernel_size[0] != kernel_size[1]
                or kernel_size[0] == 3
                or kernel_size[0] > 4
            ):
                raise Exception(
                    "Kernel size is limited to 1x1, 2x2 or 4x4 for AvgPool2D layer."
                )

        if (
            len(kernel_size) > 2
            or len(stride) > 2
            or len(padding) > 2
            or len(input_dimension) > 2
        ):
            raise Exception(
                "We expect input dimension kernel, stride and padding to be 2D elements, i.e.,"
                "to have only two positions: x and y."
            )

        if kernel_size[0] > 16 or kernel_size[1] > 16:
            raise Exception("Kernel size is limited to, at most, 16x16.")

        if output_feature_size > 1024:
            raise Exception("Output feature size is limited to, at most, 1024.")

        if not self.check_stride():
            raise Warning("Kernel stride can be 1, 2, 4 or 8 and, at most, 8x8.")

    def calculate_total_memory(self):
        pass

    def calculate_kernel_memory(self):
        pass

    def calculate_neuron_memory(self):
        pass

    def check_stride(self):
        pass

    def verify_combined_memories(
        self, base_name: str, base_memory: int, compared_name: str, compared_memory: int
    ):
        pass
