from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential

from . import functional
from .reshape import SqueezeMixin
from .stateful_layer import StatefulLayer


class LIF(StatefulLayer):

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            state_names=["v_mem", "i_syn"] if tau_syn is not None else ["v_mem"]
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(
                torch.exp(-1.0 / torch.as_tensor(tau_mem, dtype=torch.float32))
            )
            self.alpha_syn = (
                nn.Parameter(
                    torch.exp(-1.0 / torch.as_tensor(tau_syn, dtype=torch.float32))
                )
                if tau_syn is not None
                else None
            )
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem, dtype=torch.float32))
            self.tau_syn = (
                nn.Parameter(torch.as_tensor(tau_syn, dtype=torch.float32))
                if tau_syn is not None
                else None
            )
        self.spike_fn = spike_fn
        self.reset_fn = reset_fn
        self.surrogate_grad_fn = surrogate_grad_fn
        self.train_alphas = train_alphas
        self.norm_input = norm_input
        self.record_states = record_states
        self.min_v_mem = (
            nn.Parameter(torch.as_tensor(min_v_mem), requires_grad=False)
            if min_v_mem is not None
            else None
        )
        self.spike_threshold = (
            nn.Parameter(torch.as_tensor(spike_threshold), requires_grad=False)
            if spike_threshold is not None
            else None
        )
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self) -> torch.Tensor:
        pass

    @property
    def alpha_syn_calculated(self) -> torch.Tensor:
        pass

    @property
    def tau_mem_calculated(self) -> torch.Tensor:
        pass

    @property
    def tau_syn_calculated(self) -> torch.Tensor:
        pass

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def shape(self):
        pass

    @property
    def _param_dict(self) -> dict:
        pass


class LIFRecurrent(LIF):

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
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

    @property
    def _param_dict(self) -> dict:
        pass


class LIFSqueeze(LIF, SqueezeMixin):

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
