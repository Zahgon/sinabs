from functools import partial
from typing import Optional, Tuple, Union

import nir
import nirtorch
import numpy as np
import torch
from torch import nn

import sinabs.layers as sl


def _as_pair(x) -> Tuple[int, int]:
    pass


def _import_sinabs_module(
    node: nir.NIRNode, batch_size: int, num_timesteps: int
) -> torch.nn.Module:
    pass


def from_nir(
    node: nir.NIRNode, batch_size: int = None, num_timesteps: int = None
) -> torch.nn.Module:
    """Load a sinabs model from an NIR model.

    Args:
        node (nir.NIRNode): An NIR node/graph of the model
        batch_size (int, optional): batch size of the data that is expected to be fed to the model.Defaults to None.
        num_timesteps (int, optional): Number of time steps per data sample. Defaults to None.

    NOTE:
        `batch_size` or `num_timesteps` has to be specified for the sinabs model to be instantiated correctly.

    Returns:
        torch.nn.Module: Returns a sinabs model that is equivalent to the NIR graph specified.
    """
    return nirtorch.load(
        node,
        partial(
            _import_sinabs_module, batch_size=batch_size, num_timesteps=num_timesteps
        ),
    )


def _extend_to_shape(x: Union[torch.Tensor, float], shape: Tuple) -> torch.Tensor:
    pass


def _extract_sinabs_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    pass


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "model"
) -> nir.NIRNode:
    """Generate a NIRGraph given a sinabs model.

    Args:
        module (torch.nn.Module): The sinabs model to be converted to NIR graph
        sample_data (torch.Tensor): A sample data that can be used to extract various shapes and internal states.
        model_name (str, optional): The name of the top level model. Defaults to "model".

    Returns:
        nir.NIRNode: Returns the equivalent NIR object.
    """
    return nirtorch.extract_nir_graph(
        module,
        _extract_sinabs_module,
        sample_data,
        model_name=model_name,
        ignore_dims=[0],
    )
