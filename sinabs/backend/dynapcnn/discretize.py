from copy import deepcopy
from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn

import sinabs.layers as sl

DYNAPCNN_WEIGHT_PRECISION_BITS = 8
DYNAPCNN_STATE_PRECISION_BITS = 16


def discretize_conv_spike(
    conv_lyr: nn.Conv2d, spike_lyr: sl.IAF, to_int: bool = True
) -> Tuple[nn.Conv2d, sl.IAF]:
    pass


def discretize_conv_spike_(
    conv_lyr: nn.Conv2d, spike_lyr: sl.IAF, to_int: bool = True
) -> Tuple[nn.Conv2d, sl.IAF]:
    pass


def discretize_conv(
    layer: nn.Conv2d,
    spk_thr: float,
    spk_thr_low: float,
    spk_state: Optional[torch.Tensor] = None,
    to_int: bool = True,
) -> nn.Conv2d:
    pass


def discretize_conv_(
    layer: nn.Conv2d,
    spk_thr: float,
    spk_thr_low: float,
    spk_state: Optional[torch.Tensor] = None,
    to_int: bool = True,
) -> nn.Conv2d:
    pass


def discretize_spk(
    layer: sl.IAF,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
) -> sl.IAF:
    pass


def discretize_spk_(
    layer: sl.IAF,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
) -> sl.IAF:
    pass


def _discretize_conv_spk_(
    conv_lyr: Optional[nn.Conv2d] = None,
    spike_lyr: Optional[sl.IAF] = None,
    spk_thr: Optional[float] = None,
    spk_thr_low: Optional[float] = None,
    spk_state: Optional[torch.Tensor] = None,
    conv_weight: Optional[torch.Tensor] = None,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
) -> Tuple[nn.Conv2d, sl.IAF]:
    pass


def determine_discretization_scale(obj: torch.Tensor, bit_precision: int) -> float:
    pass


def discretize_tensor(
    obj: torch.Tensor, scaling: float, to_int: bool = True
) -> torch.Tensor:
    pass


def discretize_scalar(obj: float, scaling: float) -> int:
    pass
