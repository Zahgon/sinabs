from typing import Callable, Optional

import torch


def lif_forward_single(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: Optional[float],
    norm_input: bool,
):
    pass


def lif_forward(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: float,
    norm_input: bool,
    record_states: bool = False,
):
    pass


def lif_recurrent(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: Optional[float],
    norm_input: bool,
    rec_connect: torch.nn.Module,
    record_states: bool = False,
):
    pass
