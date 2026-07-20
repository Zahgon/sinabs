from typing import Callable

import torch


def alif_forward_single(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: float,
    b0: float,
    norm_input: bool,
):
    pass


def alif_forward(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: float,
    b0: float,
    norm_input: bool,
    record_states: bool = False,
):
    pass


def alif_recurrent(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: float,
    rec_connect: torch.nn.Module,
    b0: float,
    norm_input: bool,
    record_states: bool = False,
):
    pass
