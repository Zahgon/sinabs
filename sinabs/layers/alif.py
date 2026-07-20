from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from sinabs.activation import MembraneSubtract, SingleExponential, SingleSpike

from . import functional
from .reshape import SqueezeMixin
from .stateful_layer import StatefulLayer


class ALIF(StatefulLayer):

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = SingleSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            state_names=(
                ["v_mem", "i_syn", "b", "spike_threshold"]
                if tau_syn is not None
                else ["v_mem", "b", "spike_threshold"]
            )
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_mem)))
            self.alpha_adapt = nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_adapt)))
            self.alpha_syn = (
                nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_syn)))
                if tau_syn is not None
                else None
            )
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem))
            self.tau_adapt = nn.Parameter(torch.as_tensor(tau_adapt))
            self.tau_syn = nn.Parameter(torch.as_tensor(tau_syn)) if tau_syn else None
        self.adapt_scale = adapt_scale
        self.b0 = spike_threshold
        self.spike_fn = spike_fn
        self.reset_fn = reset_fn
        self.surrogate_grad_fn = surrogate_grad_fn
        self.min_v_mem = min_v_mem
        self.train_alphas = train_alphas
        self.norm_input = norm_input
        self.record_states = record_states
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        pass

    @property
    def alpha_adapt_calculated(self):
        pass

    @property
    def alpha_syn_calculated(self):
        pass

    def forward(self, input_data: torch.Tensor):
        pass

    @property
    def shape(self):
        pass

    @property
    def _param_dict(self) -> dict:
        pass


class ALIFRecurrent(ALIF):

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = SingleSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_adapt=tau_adapt,
            tau_syn=tau_syn,
            adapt_scale=adapt_scale,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            train_alphas=train_alphas,
            norm_input=norm_input,
            record_states=record_states,
        )
        self.rec_connect = rec_connect

    def forward(self, input_data: torch.Tensor):
        pass


class ALIFSqueeze(ALIF, SqueezeMixin):

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
