import math
import os
from itertools import groupby
from multiprocessing import Process
from typing import Dict, List, Tuple

import numpy as np
import samna
import samnagui
import torch

from .utils import parse_device_id, standardize_device_id

device_types = {
    "speck2e": "Speck2eTestBoard",  # with a capital B for board
    "speck2edevkit": "Speck2eDevKit",
    "speck2fmodule": "Speck2fModuleDevKit",
    "speck2fdevkit": "Speck2fDevKit",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
}

device_type_map = {v: k for (k, v) in device_types.items()}
device_map = {}


def enable_timestamps(
    device_id: str,
) -> None:
    pass


def disable_timestamps(
    device_id: str,
) -> None:
    pass


def reset_timestamps(
    device_id: str,
) -> None:
    pass


def events_to_xytp(event_list: List, layer: int) -> np.array:
    pass


def get_device_map() -> Dict:
    """
    Returns
    -------

        dict(str: samna.device.DeviceInfo)
        Returns a dict of device name and device identifier.
    """

    def sort_devices(devices: List) -> List:
        devices.sort(key=lambda x: x.usb_device_address)
        return devices

    devices = samna.device.get_all_devices()
    device_groups = groupby(devices, lambda x: x.device_type_name)
    device_groups = {
        device_type_map[k]: sort_devices(list(v))
        for k, v in device_groups
        if k in device_type_map
    }
    for dev_type, dev_list in device_groups.items():
        for i, dev in enumerate(dev_list):
            device_map[f"{dev_type}:{i}"] = dev
    return device_map


def is_device_type(dev_info: samna.device.DeviceInfo, dev_type: str) -> bool:
    pass


def discover_device(device_id: str):
    pass


def open_device(device_id: str):
    """Open device function.

    Args:
        device_id: device_name:device_id pair given as a string

    Returns:
        Device handle received from samna.
    """
    device_id = standardize_device_id(device_id=device_id)
    device_map = get_device_map()
    try:
        device_info = device_map[device_id]
    except KeyError:
        msg = f"Device {device_id} has not been found. Make sure it is connected."
        if device_map:
            msg += "The following devices are available:\n" + "\n".join(device_map)
        raise IOError(msg)
    device_handle = samna.device.open_device(device_info)

    if device_handle is not None:
        return device_handle
    else:
        raise IOError("The connection to the device cannot be established.")


def close_device(device_id: str):
    """Close a device by device identifier.

    Args:
        device_id: device_name:device_id pair given as a string.
            speck2fdevkit:0 or speck2edevkit:0 or speck2fdevkit:1 or ...
    """
    device_id = standardize_device_id(device_id=device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    print(f"Closing device: {device_id}")
    samna.device.close_device(device_handle)


def launch_visualizer(
    receiver_endpoint: str,
    width_proportion: float = 0.6,
    height_proportion: float = 0.6,
    disjoint_process: bool = True,
):
    pass


def calculate_neuron_address(
    x: int, y: int, c: int, feature_map_size: Tuple[int, int, int]
) -> int:
    """Calculate the neuron address on the devkit. This function is designed for ReadNeuronValue
    event to help the user check the neuron value of the SNN on the devkit.

    Args:
        x: x coordinate of the neuron
        y: y coordinate of the neuron
        c: channel index of the neuron
        feature_map_size: Tuple[int, int, int] the size of the feature map
            [channel, height, width]

    Returns:
        neuron_address: int
    """
    channel, height, width = feature_map_size
    x_bits = math.ceil(math.log2(width))
    y_bits = math.ceil(math.log2(height))
    channel_bits = math.ceil(math.log2(channel))
    assert (
        x_bits + y_bits + channel_bits <= 18
    ), "Bits overflow! Check if your input arguments are correct!"

    x_shift_bits = channel_bits
    y_shift_bits = channel_bits + y_bits
    y_address = y << y_shift_bits
    x_address = x << x_shift_bits
    c_address = c

    neuron_address = y_address | x_address | c_address

    return neuron_address


def neuron_address_to_cxy(
    address: int, feature_map_size: Tuple[int, int, int]
) -> Tuple:
    """Calculate the c, x, y, coordinate of a neuron when the address of the
    NeuronValue event is given.

    Args:
        address (int): the neuron address of the NeuronValue event
        feature_map_size: Tuple[int, int, int] the size of the feature map
            [channel, height, width]

    Returns:
        neuron_cxy: Tuple[int, int, int] the [channel, x, y] of the neuron
    """
    channel, height, width = feature_map_size
    x_bits = math.ceil(math.log2(width))
    y_bits = math.ceil(math.log2(height))
    channel_bits = math.ceil(math.log2(channel))
    assert (
        x_bits + y_bits + channel_bits <= 18
    ), "Bits overflow! Check if your input arguments are correct!"

    x_shift_bits = channel_bits
    y_shift_bits = channel_bits + y_bits

    y = address >> y_shift_bits
    x = (address >> x_shift_bits) & (2**x_bits - 1)
    c = address & (2**channel_bits - 1)

    return c, x, y
