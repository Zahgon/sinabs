from typing import Callable, Optional

import torch
import torch.nn as nn


class Repeat(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        pass

    def __repr__(self):
        return "Repeated " + self.module.__repr__()


class FlattenTime(nn.Flatten):

    def __init__(self):
        super().__init__(start_dim=0, end_dim=1)


class UnflattenTime(nn.Module):

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, x):
        pass


class SqueezeMixin:

    def squeeze_init(self, batch_size: Optional[int], num_timesteps: Optional[int]):
        pass

    def squeeze_forward(self, input_data: torch.Tensor, forward_method: Callable):
        pass

    def squeeze_param_dict(self, param_dict: dict) -> dict:
        pass
