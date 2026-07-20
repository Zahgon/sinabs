import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import pylab


from .layers import StatefulLayer
from .synopcounter import SNNAnalyzer
from .utils import get_activations, get_network_activations

ArrayLike = Union[np.ndarray, List, Tuple]


class Network(torch.nn.Module):

    def __init__(
        self,
        analog_model=None,
        spiking_model=None,
        input_shape: Optional[ArrayLike] = None,
        synops: bool = False,
        batch_size: int = 1,
        num_timesteps: int = 1,
    ):
        super().__init__()
        self.spiking_model: nn.Module = spiking_model
        self.analog_model: nn.Module = analog_model
        self.input_shape = input_shape

        self.synops = synops
        if synops:
            self.synops_counter = SNNAnalyzer(self.spiking_model)

        if input_shape is not None and spiking_model is not None:
            self._compute_shapes(
                input_shape, batch_size=batch_size, num_timesteps=num_timesteps
            )

    @property
    def layers(self):
        pass

    def _compute_shapes(self, input_shape, batch_size=1, num_timesteps=1):
        pass

    def forward(self, tsrInput) -> torch.Tensor:
        pass

    def compare_activations(
        self,
        data,
        name_list: Optional[ArrayLike] = None,
        compute_rate: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        pass

    def plot_comparison(
        self, data, name_list: Optional[ArrayLike] = None, compute_rate=False
    ):
        pass

    def reset_states(
        self,
        randomize: bool = False,
        value_ranges: Optional[List[Dict[str, Tuple[float, float]]]] = None,
    ):
        """Reset all neuron states in the submodules.

        Args:
            randomize (bool): If true, reset the states between a range
                provided. Else, the states are reset to zero.
            value_ranges (Optional[List[Dict[str, Tuple[float, float]]]]): A
                list of value_range dictionaries with the same length as the
                total stateful layers in the module. Each dictionary is a key
                value pair: buffer_name -> (min, max) for each state that needs
                to be reset.
                The states are reset with a uniform distribution between the min
                and max values specified. Any state with an undefined key in
                this dictionary will be reset between 0 and 1.
                This parameter is only used if randomize is set to true.
        """

        if value_ranges:
            num_stateful_layers = len(
                [None for mod in self.modules() if isinstance(mod, StatefulLayer)]
            )
            if len(value_ranges) != num_stateful_layers:
                raise TypeError(
                    "The number of entries in value_ranges does not match the number of stateful sub modules"
                )
        i = 0
        for lyr in self.modules():
            if isinstance(lyr, StatefulLayer):
                if value_ranges is None:
                    vr = None
                else:
                    vr = value_ranges[i]
                    i += 1
                lyr.reset_states(randomize=randomize, value_ranges=vr)

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def get_synops(self, num_evs_in=None) -> dict:
        pass


