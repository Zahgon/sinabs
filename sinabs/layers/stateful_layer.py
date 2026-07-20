import random
from typing import Dict, List, Optional, Tuple

import torch


class StatefulLayer(torch.nn.Module):

    def __init__(self, state_names: List[str]):
        super().__init__()

        for state_name in state_names:
            self.register_buffer(state_name, torch.zeros((0)))

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def forward(self, *args, **kwargs):
        """
        Not implemented - You need to implement a forward method in child class
        """
        raise NotImplementedError(
            "No forward method has been implemented for this class"
        )

    def is_state_initialised(self) -> bool:
        """Checks if buffers are of shape 0 and returns True only if none of them are."""
        for buffer in self.buffers():
            if buffer.shape == torch.Size([0]):
                return False
        return True

    def state_has_shape(self, shape) -> bool:
        pass

    def handle_state_batch_size_mismatch(self, new_batch_size: int):
        pass

    def has_trailing_dimension(self, trailing_dim: Tuple[int, int, int]) -> bool:
        pass

    def init_state_with_shape(self, shape, randomize: bool = False) -> None:
        pass

    def reset_states(
        self,
        randomize: bool = False,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Reset the state/buffers in a layer.

        Args:
            randomize: If true, reset the states between a range provided.
                Else, the states are reset to zero.
            value_ranges: A dictionary of key value pairs: buffer_name -> (min,
                max) for each state that needs to be reset. The states are
                reset with a uniform distribution between the min and max
                values specified. Any state with an undefined key in this
                dictionary will be reset between 0 and 1. This parameter is
                only used if randomize is set to true.

        .. note:: If you would like to reset the state with a custom distribution, you can do this individually for each parameter as follows::

            layer.<state_name>.data = <your desired data>

            layer.<state_name>.detach_()
        """
        if self.is_state_initialised():
            for name, buffer in self.named_buffers():
                if randomize:
                    if value_ranges and name in value_ranges:
                        min_value, max_value = value_ranges[name]
                    else:
                        min_value, max_value = (0.0, 1.0)
                    torch.nn.init.uniform_(buffer)
                    buffer.data = buffer * (max_value - min_value) + min_value
                else:
                    buffer.zero_()
                buffer.detach_()

    def __repr__(self):
        param_strings = [
            f"{key}={value}"
            for key, value in self._param_dict.items()
            if key
            in [
                "tau_mem",
                "tau_syn",
                "tau_adapt",
                "adapt_scale",
                "spike_threshold",
                "min_v_mem",
                "norm_input",
                "batch_size",
                "num_timesteps",
            ]
            and value is not None
        ]
        param_strings = ", ".join(param_strings)
        return f"{self.__class__.__name__}({param_strings})"

    def __deepcopy__(self, memo=None):
        copy = self.__class__(**self._param_dict)

        for name, param in self.named_parameters():
            new_inst_param = getattr(copy, name)
            new_inst_param.data = param.data.clone()
        for name, buffer in self.named_buffers():
            new_inst_buffer = getattr(copy, name)
            new_inst_buffer.data = buffer.data.clone()  # Copy parameters

        return copy

    @property
    def _param_dict(self) -> dict:
        pass

    @property
    def arg_dict(self) -> dict:
        pass

    @property
    def does_spike(self) -> bool:
        pass
