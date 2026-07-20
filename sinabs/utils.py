from typing import Iterable, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn

import sinabs
from .validate_memory_speck import ValidateMapping


def get_new_index(existing_indices: Sequence) -> int:
    """Get a new index that is not yet part of a Sequence of existing indices

    Example:
    `get_new_index([0,1,2,3])`: `4`
    `get_new_index([0,1,3])`: `2`

    Args:
        existing_indices: Sequence of indices

    Returns:
        Smallest positive integer number (starting from 0) that is not yet in
            `existing_indices`.
    """
    existing_indices = set(existing_indices)
    possible_indices = range(len(existing_indices) + 1)
    unused_indices = existing_indices.symmetric_difference(possible_indices)
    return min(unused_indices)


def reset_states(model: nn.Module) -> None:
    """Helper function to recursively reset all states of spiking layers within the model.

    Args:
        model: The torch module
    """
    for layer in model.children():
        if len(list(layer.children())):
            reset_states(layer)
        elif isinstance(layer, sinabs.layers.StatefulLayer):
            layer.reset_states()


def zero_grad(model: nn.Module) -> None:
    pass


def get_activations(torchanalog_model, tsrData, name_list=None):
    pass


def get_network_activations(
    model: nn.Module, inp, name_list: List = None, bRate: bool = False
) -> List[np.ndarray]:
    pass


def normalize_weights(
    ann: nn.Module,
    sample_data: torch.Tensor,
    output_layers: List[str],
    param_layers: List[str],
    percentile: float = 99,
):
    """Rescale the weights of the network, such that the activity of each specified layer is
    normalized.

    The method implemented here roughly follows the paper:
    `Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification` by Rueckauer et al.
    https://www.frontiersin.org/article/10.3389/fnins.2017.00682

    Args:
         ann: Torch module
         sample_data: Input data to normalize the network with
         output_layers: List of layers to verify activity of normalization.
            Typically this is a relu layer.
         param_layers: List of layers whose parameters preceed `output_layers`
         percentile: A number between 0 and 100 to determine activity to be normalized by
            where a 100 corresponds to the max activity of the network. Defaults to 99.
    """
    output_data = []

    def save_data(lyr, input, output):
        pass

    named_layers = dict(ann.named_children())

    for i in range(len(output_layers)):
        param_layer = named_layers[param_layers[i]]
        output_layer = named_layers[output_layers[i]]

        handle = output_layer.register_forward_hook(save_data)

        with torch.no_grad():
            _ = ann(sample_data)

            max_lyr_out = np.percentile(output_data[-1].cpu().numpy(), percentile)

            for p in param_layer.parameters():
                p.data *= 1 / max_lyr_out

        output_data.clear()
        handle.remove()


def set_batch_size(model: nn.Module, batch_size: int):
    pass


def get_batch_size(model: nn.Module) -> int:
    pass


def get_num_timesteps(model: nn.Module) -> int:
    pass


def get_smallest_compatible_time_dimension(model: nn.Module) -> int:
    pass


def expand_to_pair(value) -> Tuple[int, int]:
    """Expand a given value to a pair (tuple) if an int is passed.

    Args:
        value (int):
    Returns:
        pair: (int, int)
    """
    return (value, value) if isinstance(value, int) else value


T = TypeVar("T")


def collapse_pair(pair: Union[Iterable[T], T]) -> T:
    """Collapse an iterable of equal elements by returning only the first

    Args:
        pair: Iterable. All elements should be the same.

    Returns:
        First item of `pair`. If `pair` is not iterable it will return `pair` itself.

    Raises:
        ValueError if not all elements in `pair` are equal.
    """
    if isinstance(pair, Iterable):
        items = [x for x in pair]
        if any(x != items[0] for x in items):
            raise ValueError("All elements of `pair` must be the same")
        return items[0]
    else:
        return pair


def validate_memory_mapping_speck(
    input_feature_size: int,
    output_feature_size: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    input_dimension: Tuple[int, int] = [64, 64],
    conv_2d: bool = True,
):
    pass
