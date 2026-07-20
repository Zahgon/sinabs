import time
from typing import Dict, List, Tuple, Union
from warnings import warn

import numpy as np
import samna
import torch.nn as nn
from samna.specksim.nodes import SpecksimConvolutionalFilterNode as ConvFilter
from samna.specksim.nodes import SpecksimIAFFilterNode as IAFFilter
from samna.specksim.nodes import SpecksimSumPoolingFilterNode as SumPoolFilter

import sinabs.layers as sl
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer


def to_tuple(x):
    pass


def convert_linear_to_convolutional(
    layer: nn.Linear, input_shape: Tuple[int, int, int]
) -> nn.Conv2d:
    pass


def convert_convolutional_layer(
    layer: nn.Conv2d, input_shape: Tuple[int, int, int], weight_scale: float = 1.0
) -> Tuple[ConvFilter, Tuple[int, int, int]]:
    pass


def convert_pooling_layer(
    layer: Union[nn.AvgPool2d, sl.SumPool2d], input_shape: Tuple[int, int, int]
) -> Tuple[SumPoolFilter, Tuple[int, int, int]]:
    pass


def convert_iaf_layer(
    layer: Union[sl.IAF, sl.IAFSqueeze], input_shape: Tuple[int, int, int]
) -> Tuple[IAFFilter, Tuple[int, int, int]]:
    pass


def calculate_weight_scale(layer: nn.AvgPool2d):
    pass


def from_sequential(
    network: nn.Sequential, input_shape: Tuple[int, int, int]
) -> "SpecksimNetwork":
    pass


class SpecksimNetwork:
    output_dtype = np.dtype(
        [("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32)]
    )

    def __init__(
        self,
        graph: samna.graph.EventFilterGraph,
        graph_members: List["SamnaFilterNode"],  # noqa: F821
        initial_sleep_duration: float = 1.0,
        subsequent_sleep_duration: float = 0.1,
    ):
        """Specksim simulation container object.

        Args:
            graph (samna.graph.EventFilterGraph): A samna graph that contains the network layers as samna filters.
            graph_members (List["SamnaFilterNode"]): A list of samna filters.
            initial_sleep_duration (float): Sleep between writing and reading from the samna graph structure. This is
            needed because the graph runs on a separate thread.
            subsequent_sleep_duration (float): In order to not drop any events, we can sleep for more time.
        """
        self.network: samna.graph.EventFilterGraph = graph
        self.members = graph_members
        self.initial_sleep_duration = initial_sleep_duration
        self.subsequent_sleep_duration = subsequent_sleep_duration

        self.monitors: Dict[int, Dict[str, List]] = {}

    def forward(self, xytp: np.record) -> np.record:
        pass

    def __call__(self, xytp: np.record) -> np.record:
        return self.forward(xytp)

    def reset_states(self):
        """
        Reset the states of every spiking layer in the network to 0.
        """
        for member in self.members:
            if isinstance(member, IAFFilter):
                member.get_layer().reset_states()

    def get_nth_spiking_layer(self, spike_layer_number: int) -> IAFFilter:
        pass

    def add_monitor(self, spike_layer_number: int):
        pass

    def add_monitors(self, spike_layer_numbers: List[int]):
        pass

    def read_monitor(self, spike_layer_number: int) -> np.record:
        pass

    def read_monitors(self, spike_layer_numbers: List[int]) -> Dict[int, np.record]:
        pass

    def read_all_monitors(self):
        pass

    def read_spiking_layer_states(
        self, spike_layer_number: int
    ) -> List[List[List[int]]]:
        pass

    def clear_monitors(self):
        pass

    @staticmethod
    def xytp_to_specksim_spikes(xytp: np.record) -> List[samna.specksim.events.Spike]:
        pass

    @staticmethod
    def specksim_spikes_to_xytp(
        spikes: List[samna.specksim.events.Spike], output_dtype: np.dtype
    ) -> np.record:
        pass
