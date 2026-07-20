from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from torch import nn

from sinabs.layers import SqueezeMixin, StatefulLayer


def _extract_single_input(input_data: List[Any]) -> Any:
    pass


def conv_connection_map(
    layer: nn.Conv2d,
    input_shape: torch.Size,
    output_shape: torch.Size,
    device: Union[None, torch.device, str] = None,
) -> torch.Tensor:
    pass


def get_hook_data_dict(module: nn.Module) -> Dict:
    pass


def input_diff_hook(
    module: Union[nn.Conv2d, nn.Linear],
    input_: List[torch.Tensor],
    output: torch.Tensor,
):
    pass


def firing_rate_hook(module: StatefulLayer, input_: Any, output: torch.Tensor):
    pass


def firing_rate_per_neuron_hook(
    module: StatefulLayer, input_: Any, output: torch.Tensor
):
    pass


def conv_layer_synops_hook(
    module: nn.Conv2d, input_: List[torch.Tensor], output: torch.Tensor
):
    pass


def linear_layer_synops_hook(
    module: nn.Linear, input_: List[torch.Tensor], output: torch.Tensor
):
    pass


@dataclass
class ModelSynopsHook:

    dt: Optional[float] = None

    def __call__(self, module: nn.Sequential, input_: Any, output: Any):
        """Forward call of the synops model hook. Should not be called manually but only by PyTorch
        during a forward pass.

        Args:
            module: A torch.nn.Sequential
            input_: List of inputs to the module.
            output: The module output.
        Effect:
            If `module` does not already have a `hook_data` attribute, it
            will be added and synaptic operations will be calculated and logged
            for all layers that have a layer-level synops hook registered.
        """
        module_data = get_hook_data_dict(module)
        module_data["total_synops_per_timestep"] = 0.0
        module_data["synops_per_timestep"] = dict()
        if self.dt is not None:
            module_data["total_synops_per_second"] = 0.0
            module_data["synops_per_second"] = dict()

        scale_factors = []
        for lyr_idx, lyr in enumerate(module):
            if isinstance(lyr, nn.AvgPool2d):
                if lyr.kernel_size != lyr.stride:
                    warn(
                        "In order for the Synops counter to work accurately the pooling "
                        f"layers kernel size should match their strides. At the moment at layer {lyr_idx}, "
                        f"the kernel_size = {lyr.kernel_size}, the stride = {lyr.stride}."
                    )
                ks = lyr.kernel_size
                scaling = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
                scale_factors.append(scaling)
            if hasattr(lyr, "weight"):
                if (
                    hasattr(lyr, "hook_data")
                    and "layer_synops_per_timestep" in lyr.hook_data
                ):
                    layer_data = lyr.hook_data
                    scaling = reduce(lambda x, y: x * y, scale_factors, 1)
                    synops = layer_data["layer_synops_per_timestep"] * scaling
                    layer_data["synops_per_timestep"] = synops
                    module_data["synops_per_timestep"][lyr_idx] = synops
                    module_data["total_synops_per_timestep"] += synops
                    if self.dt is not None:
                        synops_per_sec = layer_data["synops_per_timestep"] / self.dt
                        layer_data["synops_per_second"] = synops_per_sec
                        module_data["synops_per_second"][lyr_idx] = synops_per_sec
                        module_data["total_synops_per_second"] += synops_per_sec

                scale_factors = []


def register_synops_hooks(module: nn.Sequential, dt: Optional[float] = None):
    pass
