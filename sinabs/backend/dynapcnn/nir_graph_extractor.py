from copy import deepcopy
from pprint import pformat
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import nirtorch
import torch
import torch.nn as nn

from sinabs import layers as sl
from sinabs.utils import get_new_index

from .connectivity_specs import (
    LAYER_TYPES_WITH_MULTIPLE_INPUTS,
    LAYER_TYPES_WITH_MULTIPLE_OUTPUTS,
    SupportedNodeTypes,
)
from .dvs_layer import DVSLayer
from .dynapcnn_layer_utils import construct_dynapcnnlayers_from_mapper
from .dynapcnnnetwork_module import DynapcnnNetworkModule
from .exceptions import InvalidGraphStructure, UnsupportedLayerType
from .sinabs_edges_handler import (
    collect_dynapcnn_layer_info,
    fix_dvs_module_edges,
    handle_batchnorm_nodes,
)
from .utils import Edge, topological_sorting
from warnings import warn

try:
    from nirtorch.graph import TorchGraph
except ImportError:
    from nirtorch.graph import Graph as TorchGraph


class GraphExtractor:
    def __init__(
        self,
        spiking_model: nn.Module,
        dummy_input: torch.tensor,
        dvs_input: Optional[bool] = None,
        ignore_node_types: Optional[Iterable[Type]] = None,
    ):
        """Class implementing the extraction of the computational graph from `spiking_model`, where
        each node represents a layer in the model and the list of edges represents how the data flow between
        the layers.

        Args:
            spiking_model (nn.Module): a sinabs-compatible spiking network.
            dummy_input (torch.tensor): an input sample to be fed through
                the model to acquire both the computational graph (via
                `nirtorch`) and the I/O shapes of each node. Its a 4-D shape
                with `(batch, channels, heigh, width)`.
            dvs_input (bool): optional (default as `None`). Whether or not the
                model should start with a `DVSLayer`.
            ignore_node_types (iterable of types): Node types that should be
                ignored completely from the graph. This can include, for
                instance, `nn.Dropout2d`, which otherwise can result in wrongly
                inferred graph structures by NIRTorch. Types such as
                `nn.Flatten`, or sinabs `Merge` should not be included here, as
                they are needed to properly handle graph structure and
                metadata. They can be removed after instantiation with
                `remove_nodes_by_class`.

        Attributes:
            edges (set of 2-tuples of integers): Tuples describing the
                connections between layers in `spiking_model`. Each layer
                (node) is identified by a unique integer ID.
            name_2_index_map (dict): Keys are original variable names of layers
                in `spiking_model`. Values are unique integer IDs.
            entry_nodes (set of ints): IDs of nodes acting as entry points for
                the network, i.e. receiving external input.
            indx_2_module_map (dict): Map from layer ID to the corresponding
                nn.Module instance.
            nodes_io_shapes (dict): Map from node ID to dict containing node's
                in- and output shapes.
            dvs_input (bool): optional (default as `None`). Whether or not the
                model should start with a `DVSLayer`.
            ignore_node_types (iterable of types): Node types that should be
                ignored completely from the graph. This can include, for
                instance, `nn.Dropout2d`, which otherwise can result in wrongly
                inferred graph structures by NIRTorch. Types such as
                `nn.Flatten`, or sinabs `Merge` should not be included here, as
                they are needed to properly handle graph structure and
                metadata. They can be removed after instantiation with
                `remove_nodes_by_class`.
        """

        original_state = {
            n: b.detach().clone() for n, b in spiking_model.named_buffers()
        }

        self._edges = set()
        if isinstance(spiking_model, nn.Sequential) and len(spiking_model) == 0:
            self._name_2_indx_map = dict()
            self._edges = set()
            original_state = {}
        else:
            nir_graph = nirtorch.graph.extract_torch_graph(
                spiking_model, dummy_input, model_name=None
            ).ignore_tensors()

            if ignore_node_types is not None:
                for node_type in ignore_node_types:
                    nir_graph = nir_graph.ignore_nodes(node_type)

            self._name_2_indx_map = self._get_name_2_indx_map(nir_graph)

            self._edges = self._get_edges_from_nir(nir_graph, self._name_2_indx_map)

        self._indx_2_module_map = self._get_name2module_map(spiking_model)

        if len(self._name_2_indx_map) > 0:
            handle_batchnorm_nodes(
                self._edges, self._indx_2_module_map, self._name_2_indx_map
            )

        self._entry_nodes = self._get_entry_nodes(self._edges)

        self._handle_dvs_input(input_shape=dummy_input.shape[1:], dvs_input=dvs_input)

        self._nodes_io_shapes = self._get_nodes_io_shapes(dummy_input)

        for n, b in spiking_model.named_buffers():
            b.set_(original_state[n].clone())

        self.verify_graph_integrity()

    @property
    def dvs_layer(self) -> Union[DVSLayer, None]:
        pass

    @property
    def dvs_node_id(self) -> Union[int, None]:
        pass

    @property
    def entry_nodes(self) -> Set[int]:
        pass

    @property
    def edges(self) -> Set[Edge]:
        pass

    @property
    def has_dvs_layer(self) -> bool:
        pass

    @property
    def name_2_indx_map(self) -> Dict[str, int]:
        pass

    @property
    def nodes_io_shapes(self) -> Dict[int, Tuple[torch.Size]]:
        pass

    @property
    def sorted_nodes(self) -> List[int]:
        pass

    @property
    def indx_2_module_map(self) -> Dict[int, nn.Module]:
        pass

    def get_dynapcnn_network_module(
        self, discretize: bool = True, weight_rescaling_fn: Optional[Callable] = None
    ) -> DynapcnnNetworkModule:
        pass

    def remove_nodes_by_class(self, node_classes: Tuple[Type]):
        pass

    def get_node_io_shapes(self, node: int) -> Tuple[torch.Size, torch.Size]:
        pass

    def verify_graph_integrity(self):
        pass

    def verify_node_types(self):
        pass

    def verify_no_isolated_nodes(self):
        pass

    def _handle_dvs_input(
        self, input_shape: Tuple[int, int, int], dvs_input: Optional[bool] = None
    ):
        pass

    def _add_dvs_node(self, dvs_input_shape: Tuple[int, int, int]) -> DVSLayer:
        pass

    def _get_dvs_node_id(self) -> Union[int, None]:
        pass

    def _validate_dvs_setup(self, dvs_input_shape: Tuple[int, int, int]) -> None:
        pass

    def _get_name_2_indx_map(self, nir_graph: TorchGraph) -> Dict[str, int]:
        pass

    def _get_edges_from_nir(
        self, nir_graph: TorchGraph, name_2_indx_map: Dict[str, int]
    ) -> Set[Edge]:
        pass

    def _get_entry_nodes(self, edges: Set[Edge]) -> Set[Edge]:
        pass

    def _get_name2module_map(self, model: nn.Module) -> Dict[int, nn.Module]:
        pass

    def _update_internal_representation(self, remapped_nodes: Dict[int, int]):
        pass

    def _sort_graph_nodes(self) -> List[int]:
        pass

    def _get_nodes_io_shapes(
        self, input_dummy: torch.tensor
    ) -> Dict[int, Dict[str, torch.Size]]:
        pass

    def _find_all_sources_of_input_to(self, node: int) -> Set[int]:
        pass

    def _find_source_of_input_to(self, node: int) -> int:
        pass

    def _find_merge_arguments(self, node: int) -> Edge:
        pass

    def _find_valid_targets(
        self, node: int, ignored_node_classes: Tuple[Type] = ()
    ) -> Set[int]:
        pass
