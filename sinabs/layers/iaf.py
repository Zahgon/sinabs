from typing import Callable, Optional

import numpy as np
import torch

from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential

from .lif import LIF, LIFRecurrent
from .reshape import SqueezeMixin


class IAF(LIF):

    def __init__(
        self,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        tau_syn: Optional[float] = None,
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=np.inf,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
            record_states=record_states,
        )
        self.tau_mem = None

    @property
    def alpha_mem_calculated(self):
        pass

    @property
    def _param_dict(self) -> dict:
        pass


class IAFRecurrent(LIFRecurrent):

    def __init__(
        self,
        rec_connect: torch.nn.Module,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        tau_syn: Optional[float] = None,
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_states: bool = False,
    ):
        super().__init__(
            rec_connect=rec_connect,
            tau_mem=np.inf,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
            record_states=record_states,
        )
        self.tau_mem = None

    @property
    def alpha_mem_calculated(self):
        pass

    @property
    def _param_dict(self) -> dict:
        pass


class IAFSqueeze(IAF, SqueezeMixin):

    def __init__(
        self,
        batch_size=None,
        num_timesteps=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def _param_dict(self) -> dict:
        pass
