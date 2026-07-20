
from typing import Deque, Dict, List, Optional, Set, Tuple, Type, Union

from torch import Size, nn

from sinabs.layers import SumPool2d

from .connectivity_specs import VALID_SINABS_EDGE_TYPES
from .crop2d import Crop2d
from .dvs_layer import DVSLayer
from .exceptions import (
    InvalidEdge,
    InvalidGraphStructure,
    default_invalid_structure_string,
)
from .flipdims import FlipDims
from .utils import Edge, merge_bn


def remap_edges_after_drop(
    dropped_node: int, source_of_dropped_node: int, edges: Set[Edge]
) -> Set[Edge]:
    pass


def handle_batchnorm_nodes(
    edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    name_2_indx_map: Dict[str, int],
) -> None:
    pass


def fix_dvs_module_edges(
    edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    name_2_indx_map: Dict[str, int],
    entry_nodes: Set[Edge],
) -> None:
    pass


def collect_dynapcnn_layer_info(
    indx_2_module_map: Dict[int, nn.Module],
    edges: Set[Edge],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    entry_nodes: Set[int],
) -> Tuple[Dict[int, Dict], Union[Dict, None]]:
    pass


def get_valid_edge_type(
    edge: Edge,
    layers: Dict[int, nn.Module],
    valid_edge_ids: Dict[Tuple[Type, Type], int],
) -> int:
    pass


def sort_edges_by_type(
    edges: Set[Edge], indx_2_module_map: Dict[int, Type]
) -> Dict[str, Set[Edge]]:
    pass


def init_new_dynapcnnlayer_entry(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    indx_2_module_map: Dict[int, nn.Module],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    node_2_layer_map: Dict[int, int],
    entry_nodes: Set[int],
) -> None:
    pass


def add_pooling_to_entry(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    pooling_chains: List[Deque[int]],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
) -> None:
    pass


def dvs_setup(
    edges_by_type: Dict[str, Set[Edge]],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
) -> Union[None, Dict]:
    pass


def init_dvs_entry(
    dvs_weight_edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
) -> Dict:
    pass


def init_dvs_entry_with_pooling(
    dvs_pooling_edge: Edge,
    pooling_weight_edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
) -> Dict:
    pass


def set_exit_destinations(dynapcnn_layer: Dict) -> None:
    pass


def set_neuron_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    indx_2_module_map: Dict[int, nn.Module],
) -> None:
    pass


def set_pooling_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    indx_2_module_map: Dict[int, nn.Module],
) -> None:
    pass


def trace_paths(node: int, remaining_edges: Set[Edge]) -> List[Deque[int]]:
    pass


def find_edges_by_source(edges: Set[Edge], source: int) -> Set[Edge]:
    pass


def verify_layer_info(
    dynapcnn_layer_info: Dict[int, Dict], edge_counts: Optional[Dict[str, int]] = None
):
    pass
