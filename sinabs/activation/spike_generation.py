from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch


class BackwardClass:
    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        pass


class MultiSpike(BackwardClass, torch.autograd.Function):

    required_states: List[str] = ["v_mem"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        pass


class MaxSpikeInner(BackwardClass, torch.autograd.Function):

    required_states: List[str] = ["v_mem", "max_num_spikes_per_bin"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        max_num_spikes_per_bin: Optional[int],
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        pass


@dataclass
class MaxSpike:

    max_num_spikes_per_bin: Optional[int] = None

    def apply(
        self,
        v_mem: torch.Tensor,
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        pass

    @property
    def required_states(self):
        pass


class SingleSpike(BackwardClass, torch.autograd.Function):

    required_states: List[str] = ["v_mem"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        pass
