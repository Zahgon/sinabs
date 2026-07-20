from collections import defaultdict, deque
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import warnings

import sinabs.layers as sl

from .crop2d import Crop2d
from .dvs_layer import DVSLayer
from .exceptions import InputConfigurationError

if TYPE_CHECKING:
    from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork

COMPLETELY_IGNORED_LAYER_TYPES = (nn.Identity, nn.Dropout, nn.Dropout2d)
IGNORED_LAYER_TYPES = (nn.Flatten, sl.Merge)

Edge = Tuple[int, int]  # Define edge-type alias


def parse_device_id(device_id: str) -> Tuple[str, int]:
    """Parse device id into device type and device index.

    Args:
        device_id (str): Device id typically of the form `device_type:index`.
            In case no index is specified, the default index of zero is returned.

    Returns:
        Tuple[str, int]: (device_type, index) Returns a tuple with the index and device type.
    """
    parts = device_id.split(sep=":")
    if len(parts) == 1:
        device_type = parts[0]
        index = 0
    elif len(parts) == 2:
        device_type, index = parts
    else:
        raise Exception(
            "Device id not understood. A string of form `device_type:index` expected."
        )

    return device_type, int(index)


def get_device_id(device_type: str, index: int) -> str:
    """Generate a device id string given a device type and its index.

    Args:
        device_type (str): Device type
        index (int): Device index

    Returns:
        str: A string of the form `device_type:index`
    """
    return f"{device_type}:{index}"


def standardize_device_id(device_id: str) -> str:
    """Standardize device id string.

    Args:
        device_id (str): Device id string. Could be of the form `device_type` or `device_type:index`

    Returns:
        str: Returns a sanitized device id of the form `device_type:index`
    """
    device_type, index = parse_device_id(device_id=device_id)
    return get_device_id(device_type=device_type, index=index)


def topological_sorting(edges: Set[Tuple[int, int]]) -> List[int]:
    pass


def convert_cropping2dlayer_to_crop2d(
    layer: sl.Cropping2dLayer, input_shape: Tuple[int, int]
) -> Crop2d:
    pass


WeightLayer = TypeVar("WeightLayer", nn.Linear, nn.Conv2d)


def merge_bn(
    weight_layer: WeightLayer, bn: Union[nn.BatchNorm1d, nn.BatchNorm2d]
) -> WeightLayer:
    pass


def merge_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    pass


def extend_readout_layer(model: "DynapcnnNetwork") -> "DynapcnnNetwork":
    pass


def infer_input_shape(
    snn: nn.Module, input_shape: Optional[Tuple[int, int, int]] = None
) -> Tuple[int, int, int]:
    pass
