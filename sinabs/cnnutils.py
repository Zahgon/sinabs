from typing import Tuple


def conv_output_size(image_length: int, kernel_length: int, stride: int) -> int:
    """Computes output dimension given input dimension, kernel size and stride, assumign no
    padding, *per* dimension given.

    :param image_length: int image size on one dimension
    :param kernel_length: int kernel_length size on one dimension
    :param stride: int Stride size on one dimension
    :return: int -- convolved output image size on one dimension
    """
    try:
        assert image_length >= kernel_length
    except AssertionError:
        raise Exception(
            "Image dimension {0} smaller than kernel dimension {1}".format(
                image_length, kernel_length
            )
        )
    return int((image_length - kernel_length) / stride + 1)


def compute_same_padding_size(kernel_length: int) -> (int, int):
    pass


def compute_padding(
    kernel_shape: tuple, input_shape: tuple, mode="valid"
) -> (int, int, int, int):
    pass


def infer_output_shape(torch_layer, input_shape: Tuple) -> Tuple:
    pass
