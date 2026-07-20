import socket
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import samna

from .dynapcnn_network import DynapcnnNetwork
from .io import launch_visualizer


class DynapcnnVisualizer:

    DEFAULT_LAYOUT_DS = [(0, 0, 0.5, 1), (0.5, 0, 1, 1), None, None]
    DEFAULT_LAYOUT_DSP = [(0, 0, 0.5, 0.66), (0.5, 0, 1, 0.66), None, (0, 0.66, 1, 1)]
    DEFAULT_LAYOUT_DRP = [(0, 0, 0.5, 0.66), None, (0.5, 0, 1, 0.66), (0, 0.66, 1, 1)]
    DEFAULT_LAYOUT_DSR = [(0, 0, 0.33, 1), (0.33, 0, 0.66, 1), (0.66, 0, 1, 1), None]

    DEFAULT_LAYOUT_DSRP = [
        (0, 0, 0.33, 0.66),
        (0.33, 0, 0.66, 0.66),
        (0.66, 0, 1, 0.66),
        (0, 0.66, 1, 1),
    ]

    LAYOUTS_DICT = {
        "ds": DEFAULT_LAYOUT_DS,
        "dsp": DEFAULT_LAYOUT_DSP,
        "drp": DEFAULT_LAYOUT_DRP,
        "dsr": DEFAULT_LAYOUT_DSR,
        "dsrp": DEFAULT_LAYOUT_DSRP,
    }

    def __init__(
        self,
        window_scale: Tuple[int, int] = (4, 8),
        dvs_shape: Tuple[int, int] = (128, 128),  # height, width
        add_readout_plot: bool = False,
        add_spike_count_plot: bool = True,
        add_power_monitor_plot: bool = False,
        spike_collection_interval: int = 500,
        readout_prediction_threshold: int = 10,
        readout_default_return_value: Optional[int] = None,
        readout_default_threshold_low: Optional[int] = 0,
        readout_default_threshold_high: Optional[int] = 32000,
        power_monitor_number_of_items: Optional[int] = 3,
        feature_names: Optional[List[str]] = None,
        readout_images: Optional[List[str]] = None,
        feature_count: Optional[int] = None,
        readout_node: Union[str, Callable] = "JitMajorityReadout",
        extra_arguments: Optional[Dict[str, Dict[str, any]]] = None,
    ):
        """Quick wrapper around Samna objects to get a basic dynapcnn visualizer.

        Args:
            window_scale: Tuple[int, int] (defaults to (4, 8))
                Scale of window based on a 16/9 monitor layout. (in height, width)
            dvs_shape (Tuple[int, int], optional):
                Shape of the DVS sensor in (height, width).
                Defaults to (128, 128) -- Speck sensor resolution.
            add_readout_plot: bool (defaults to False)
                If set true adds a readout plot to the GUI
                It displays an icon for the currently predicted class.
            add_spike_count_plot: bool (defaults to True)
                If set true adds a spike count plot to the GUI.
                A line chart indicating the number of spikes over time.
            add_power_monitor_plot: bool (defaults to False)
                If set true adds a power monitor plot to the GUI.
            spike_collection_interval: int (defaults to 500) (in milliseconds)
                Spike collection is done using a low-pass filter with a window size.
                This parameter sets the window size of the spike collection
            readout_prediction_threshold: int (defaults to 10)
                Defines the number of spikes needed for making a prediction.
            readout_default_return_value: Optional[int] (defaults to None)
                Defines the default prediction of the network. Usually used for `other` class in the
                network.
            readout_default_threshold_low: Optional[int] (defaults to 0)
                Default lower threshold value for `MajorityReadoutNode`
            readout_default_threshold_high: Optional[int] (defaults to int.max())
                Default higher threshold value for `MajorityReadoutNode`
            power_monitor_number_of_items: Optional[int] (defaults to 3)
                Can be set to `3` or `5`
            feature_names: Optional[List[str]] (defaults to None)
                List of feature names. If this is passed they will be displayed on the spike count plot
                as output labels
            readout_images: Optional[List[str]] (defaults to None)
                List of paths of the images to be shown in the readout plot.
                If the `feature_names` parameter is not passed the names of the images will be parsed and
                used as the spike count plot labels.
                Format of the individual file name should be of the following type.
                `classnumber`_`classlabel`.`extension`
                NOTE: Class numbers should match that of the network output channels. This is so that they
                can be sorted properly. At each operating system the behaviour in which the extraction of the
                images from a folder may differ.
                NOTE: For now only `.png` images are supported.
            feature_count: Optional[int] (defaults to None)
                If the `feature_names` and `readout_images` was passed, this is not needed. Otherwise this parameter
                should be passed, so that the GUI knows how many lines should be drawn on the `Spike Count Plot` and
                `Readout Layer Plot`.
            readout_node: str or Callable
                Can either be a string "JitMajorityReadout" or a callable that returns a samna JIT filter
                to decide on the readout prediction. Function parameters can be defined freely.
            extra_arguments: Optional[Dict[str, Dict[str, any]]] (defaults to None)
                Extra arguments that can be passed to individual plots. Available keys are:
                - `spike_count`: Arguments that can be passed to `spike_count` plot.
                - `readout`: Arguments that can be passed to `readout` plot.
                - `power_measurement`: Arguments that can be passed `power_measurement` plot.
        """
        if add_readout_plot and readout_images is None:
            raise ValueError(
                "If a readout plot is to be displayed image paths should be passed as a list."
                + "The order of the images, should match the model output."
            )

        self.window_scale = window_scale
        self.feature_names = feature_names
        self.readout_images = readout_images
        self.feature_count = feature_count
        self.dvs_shape = dvs_shape

        self.gui_type = "d"
        if add_spike_count_plot:
            self.gui_type += "s"
        if add_readout_plot:
            self.gui_type += "r"
        if add_power_monitor_plot:
            self.gui_type += "p"

        self.spike_collection_interval = spike_collection_interval

        self.readout_prediction_threshold = readout_prediction_threshold
        self.readout_default_return_value = readout_default_return_value
        self.readout_default_threshold_low = readout_default_threshold_low
        self.readout_default_threshold_high = readout_default_threshold_high
        self.readout_node = readout_node

        if power_monitor_number_of_items != 3 and power_monitor_number_of_items != 5:
            warnings.warn(
                "Power monitor number of items can be 3 ('io', 'logic', 'memory') or"
                + "5 ('io', 'logic', 'memory', 'vdd', 'vda'). Setting to 3."
            )
            power_monitor_number_of_items = 3
        self.power_monitor_number_of_items = power_monitor_number_of_items

        self.samna_visualizer_port = get_free_tcp_port()

        self.dvs_layer_id = 13

        self.extra_arguments = extra_arguments

    @staticmethod
    def parse_feature_names_from_image_names(readout_image_paths: List[str]):
        pass

    def create_visualizer_process(
        self, visualizer_endpoint: str, disjoint_process: bool = False
    ):
        pass

    def add_dvs_plot(
        self,
        shape: Tuple[int, int],
        layout: Tuple[float, float, float, float],
    ):
        pass

    def add_readout_plot(self, layout: Tuple[float, float, float, float]):
        pass

    def add_output_prediction_layer_plot():
        """
        What we want to have is something as described below:
        Plot for visualizating the chip readout layers.

        output neuron id
        ^
        |
        +-------+----------------------+
        + out0  +           x          +
        + out1  +     x                +
        + out2  +   xxxxxxxx    xxxxx  +
        +-------+----------------------+ --> time

        Where every time the readout layer has been read, if some output returns True, put an `x` there,
        denoting a prediction
        """
        raise NotImplementedError("Waiting for samna support!")

    def add_spike_count_plot(self, layout: Tuple[float, float, float, float]):
        pass

    def add_power_monitor_plot(self, layout: Tuple[int, int, int, int]):
        pass

    def create_plots(self):
        pass

    def connect(
        self, dynapcnn_network: DynapcnnNetwork, disjoint_process: bool = False
    ):
        pass

    def update_feature_count(self, dynapcnn_network: DynapcnnNetwork):
        pass

    def update_feature_names(self):
        pass

    def update_default_readout_return_value(self):
        pass

    def start(self):
        self.streamer_graph.start()

    def stop(self):
        self.streamer_graph.stop()


def get_free_tcp_port():
    pass
