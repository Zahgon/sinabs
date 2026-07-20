from typing import List, Optional, Tuple

import numpy as np
import torch

from .chips import (
    Speck2EConfigBuilder,
    Speck2FConfigBuilder,
)
from .config_builder import ConfigBuilder
from .utils import parse_device_id


class ChipFactory:
    supported_devices = {
        "speck2e": Speck2EConfigBuilder,
        "speck2edevkit": Speck2EConfigBuilder,
        "speck2fmodule": Speck2FConfigBuilder,  # Speck2fModuleDevKit
        "speck2fdevkit": Speck2FConfigBuilder,  # Speck2fDevKit
    }

    device_name: str
    device_id: int

    def __init__(self, device_str: str):
        """Factory class to access config builder and other device specific methods.

        Args:
            device_str: name of the device
        """
        self.device_name, self.device_id = parse_device_id(device_str)
        if self.device_name not in self.supported_devices:
            raise Exception(f"Builder not found for device type: {self.device_name}")

    def get_config_builder(self) -> ConfigBuilder:
        return self.supported_devices[self.device_name]()

    def raster_to_events(
        self,
        raster: torch.Tensor,
        layer,
        dt=1e-3,
        truncate: bool = False,
        delay_factor: float = 0,
    ) -> List:
        pass

    def xytp_to_events(
        self, xytp: np.ndarray, layer, reset_timestamps, delay_factor: float = 0
    ) -> List:
        pass

    def events_to_raster(
        self, events: List, dt: float = 1e-3, shape: Optional[Tuple] = None
    ) -> torch.Tensor:
        pass
