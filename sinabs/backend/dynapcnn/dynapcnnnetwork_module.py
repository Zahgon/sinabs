from pprint import pformat
from typing import Dict, List, Optional, Set, Union
from warnings import warn

import torch.nn as nn
from torch import Tensor

import sinabs.layers as sl

from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .utils import Edge, topological_sorting


class DynapcnnNetworkModule(nn.Module):

    def __init__(
        self,
        dynapcnn_layers: Dict[int, DynapcnnLayer],
        destination_map: Dict[int, List[int]],
        entry_points: Set[int],
        dvs_node_info: Optional[Dict] = None,
    ):
        super().__init__()

        self._dvs_node_info = dvs_node_info

        module_dict = {str(idx): lyr for idx, lyr in dynapcnn_layers.items()}
        self._dynapcnn_layers = nn.ModuleDict(module_dict)

        if self._dvs_node_info is not None:
            self._dvs_layer = dvs_node_info["module"]
        else:
            self._dvs_layer = None

        self._destination_map = destination_map
        self._entry_points = entry_points

        self.merge_layer = sl.Merge()

    @property
    def all_layers(self):
        pass

    @property
    def dvs_node_info(self):
        pass

    @property
    def dvs_layer(self):
        pass

    @property
    def destination_map(self):
        pass

    @property
    def dynapcnn_layers(self):
        pass

    @property
    def entry_points(self):
        pass

    @property
    def sorted_nodes(self):
        pass

    @property
    def node_source_map(self):
        pass

    def get_exit_layers(self) -> List[int]:
        """Get layers that act as exit points of the network

        Returns:
            Layer indices with at least one exit destination.
        """
        return [
            layer_idx
            for layer_idx, destinations in self.destination_map.items()
            if any(d < 0 for d in destinations)
        ]

    def get_exit_points(self) -> Dict[int, Dict]:
        pass

    def setup_dynapcnnlayer_graph(
        self, index_layers_topologically: bool = False
    ) -> None:
        pass

    def get_dynapcnnlayers_edges(self) -> Set[Edge]:
        pass

    def add_entry_points_edges(self, dcnnl_edges: Set[Edge]) -> None:
        pass

    def get_node_source_map(self, dcnnl_edges: Set[Edge]) -> Dict[int, List[int]]:
        pass

    def forward(
        self, x, return_complete: bool = False
    ) -> Union[Tensor, Dict[int, Dict[int, Tensor]]]:
        pass

    def reindex_layers(self, index_order: List[int]) -> None:
        pass

    def __repr__(self):
        return f"DVS Layer: {pformat(self.dvs_layer)}\n\nDynapCNN Layers:\n" + pformat(
            self.dynapcnn_layers
        )
