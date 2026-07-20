from typing import Optional, Union

import torch

from .lif import LIF
from .reshape import SqueezeMixin


class ExpLeak(LIF):

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        min_v_mem: Optional[float] = None,
        norm_input: bool = False,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=None,
            spike_threshold=None,
            train_alphas=train_alphas,
            min_v_mem=min_v_mem,
            shape=shape,
            spike_fn=None,
            reset_fn=None,
            surrogate_grad_fn=None,
            norm_input=norm_input,
            record_states=record_states,
        )

    @property
    def _param_dict(self) -> dict:
        pass


class ExpLeakSqueeze(ExpLeak, SqueezeMixin):

    def __init__(self, batch_size=None, num_timesteps=None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def _param_dict(self) -> dict:
        pass
