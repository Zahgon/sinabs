import time
from pprint import pformat
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import samna
import torch
import torch.nn as nn
from torch import Tensor

import sinabs
import sinabs.layers as sl

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .nir_graph_extractor import GraphExtractor
from .utils import (
    COMPLETELY_IGNORED_LAYER_TYPES,
    IGNORED_LAYER_TYPES,
    infer_input_shape,
    parse_device_id,
)
from .weight_rescaling_methods import rescale_method_1


class DynapcnnNetwork(nn.Module):
    def __init__(
        self,
        snn: nn.Module,
        input_shape: Optional[Tuple[int, int, int]] = None,
        batch_size: Optional[int] = None,
        dvs_input: Optional[bool] = None,
        discretize: bool = True,
        weight_rescaling_fn: Callable = rescale_method_1,
    ):
        """Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
        test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
        make the dynapcnn configuration and upload it to DYNAPCNN.

        Attributes:
            snn (nn.Module): a  implementing a spiking network.
            input_shape (tuple or None): a description of the input dimensions
                as `(features, height, width)`. If `None`, `snn` must contain a
                `DVSLayer` instance, from which the input shape will be inferred.
            batch_size (optional int): If `None`, will try to infer the batch
                size from the model. If int value is provided, it has to match
                the actual batch size of the model.
            dvs_input (bool): optional (default as `None`). Wether or not dynapcnn
                receive input from its DVS camera.
                If a `DVSLayer` is part of `snn`...
                ... and `dvs_input` is `False`, its `disable_pixel_array` attribute
                    will be set `True`. This means the DVS sensor will be configured
                    upon deployment but its output will not be sent as input
                ... and `dvs_input` is `None`, the `disable_pixel_array` attribute
                    of the layer will not be changed.
                ... and `dvs_input` is `True`, `disable_pixel_array` will be set
                    `False`, so that the DVS sensor data is sent to the network.
                If no `DVSLayer` is part of `snn`...
                ... and `dvs_input` is `False` or `None`, no `DVSLayer` will be added
                    and the DVS sensor will not be configured upon deployment.
                ... and `dvs_input` is `True`, a `DVSLayer` instance will be added
                    to the network, with `disable_pixel_array` set to `False`.
            discretize (bool): If `True`, discretize the parameters and thresholds.
                This is needed for uploading weights to dynapcnn. Set to `False`
                only for testing purposes.
            weight_rescaling_fn (callable): a method that handles how the re-scaling
                factor for one or more `SumPool2d` projecting to the same convolutional
                layer are combined/re-scaled before applying them.
        """
        super().__init__()

        if isinstance(snn, sinabs.Network):
            snn = snn.spiking_model

        self.dvs_input = dvs_input
        self.input_shape = infer_input_shape(snn, input_shape)
        self._layer2core_map = None

        if batch_size is None:
            batch_size = sinabs.utils.get_smallest_compatible_time_dimension(snn)
        self._graph_extractor = GraphExtractor(
            snn,
            torch.randn((batch_size, *self.input_shape)),
            self.dvs_input,
            ignore_node_types=COMPLETELY_IGNORED_LAYER_TYPES,
        )

        self._graph_extractor.remove_nodes_by_class(IGNORED_LAYER_TYPES)

        self._dynapcnn_module = self._graph_extractor.get_dynapcnn_network_module(
            discretize=discretize, weight_rescaling_fn=weight_rescaling_fn
        )
        self._dynapcnn_module.setup_dynapcnnlayer_graph(index_layers_topologically=True)

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
    def chip_layers_ordering(self):
        pass

    @property
    def dynapcnn_layers(self):
        pass

    @property
    def dynapcnn_module(self):
        pass

    @property
    def exit_layers(self):
        pass

    @property
    def exit_layer_ids(self):
        pass

    @property
    def is_deployed_on_dynapcnn_device(self):
        pass

    @property
    def layer_destination_map(self):
        pass

    @property
    def layer2core_map(self):
        pass

    @property
    def name_2_indx_map(self):
        pass

    def hw_forward(self, x):
        pass

    def forward(
        self, x, return_complete: bool = False
    ) -> Union[List["event"], Tensor, Dict[int, Dict[int, Tensor]]]:
        pass

    def parameters(self) -> list:
        """Gathers all the parameters of the network in a list. This is done by accessing the convolutional layer in each `DynapcnnLayer`,
        calling its `.parameters` method and saving it to a list.

        Note: the method assumes no biases are used.

        Returns:
            List of parameters of all convolutional layers in the `DynapcnnNetwok`.
        """
        parameters = []

        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, DynapcnnLayer):
                parameters.extend(layer.conv_layer.parameters())

        return parameters

    def memory_summary(self) -> Dict[str, Dict[int, int]]:
        """Get a summary of the network's memory requirements.

        Returns:
            A dictionary with keys kernel, neuron, bias. The values are a dicts.
            Each nested dict has as keys the indices of all dynapcnn_layers and
            as values the corresonding memory values for each layer.
        """
        summary = {key: dict() for key in ("kernel", "neuron", "bias")}

        for layer_index, layer in self.dynapcnn_layers.items():
            for key, val in layer.memory_summary().items():
                summary[key][layer_index] = val

        return summary

    def init_weights(self, init_fn: nn.init = nn.init.xavier_normal_) -> None:
        pass

    def detach_neuron_states(self) -> None:
        pass

    def to(
        self,
        device: str = "cpu",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        slow_clk_frequency: Optional[int] = None,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ):
        """Deploy model to cpu, gpu or a SynSense device.

        Note that the model parameters are only ever transferred to the device on the `to` call,
        so changing a threshold or weight of a model that is deployed will have no effect on the
        model on chip until `to` is called again.

        Args:
            device (str): cpu:0, cuda:0, speck2edevkit
            monitor_layers: None/List. A list of all layers in the module that
                you want to monitor. Indexing starts with the first non-dvs
                layer. If you want to monitor the dvs layer for eg.
                ::

                    monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                    monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                    monitor_layers = "all" # If you want to monitor all the layers
                    monitor_layers = [-1] # If you want to only monitor exit points of the network (i.e. final layers)
            config_modifier: A user configuration modifier method. This function
                can be used to make any custom changes you want to make to the configuration object.
            layer2core_map (dict or "auto"): Defines how cores on chip are
                assigned to DynapcnnLayers. If `auto`, an automated procedure
                will be used to find a valid ordering. Otherwise a dict needs
                to be passed, with DynapcnnLayer indices as keys and assigned
                core IDs as values. DynapcnnLayer indices have to match those of
                `self.dynapcnn_layers`.
            chip_layers_ordering: sequence of integers or `auto`. The order in
                which the dynapcnn layers will be used. If `auto`,an automated
                procedure will be used to find a valid ordering. A list of
                layers on the device where you want each of the model's
                DynapcnnLayers to be placed.
                The index of the core on chip to which the i-th layer in the
                model is mapped is the value of the i-th entry in the list.
                Note: This list should be the same length as the number of
                dynapcnn layers in your model.
                Note: This parameter is obsolete and should not be passed
                anymore. Use `layer2core_map` instead.

                        Note
        ----
        chip_layers_ordering and monitor_layers are used only when using synsense devices.
        For GPU or CPU usage these options are ignored.
        """
        self.device = device

        if isinstance(device, torch.device):
            self._to_device(device)

        elif isinstance(device, str):
            device_name, _ = parse_device_id(device)

            if device_name in ChipFactory.supported_devices:
                config = self.make_config(
                    layer2core_map=layer2core_map,
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

                self.samna_device = open_device(device)
                self.samna_device.get_model().apply_configuration(config)
                time.sleep(1)

                if slow_clk_frequency is not None:
                    dk_io = self.samna_device.get_io_module()
                    dk_io.set_slow_clk(True)
                    dk_io.set_slow_clk_rate(slow_clk_frequency)  # Hz

                builder = ChipFactory(device).get_config_builder()

                self.samna_input_buffer = builder.get_input_buffer()

                self.samna_output_buffer = builder.get_output_buffer()

                self.device_input_graph = samna.graph.EventFilterGraph()
                self.device_input_graph.sequential(
                    [
                        self.samna_input_buffer,
                        self.samna_device.get_model().get_sink_node(),
                    ]
                )

                self.device_output_graph = samna.graph.EventFilterGraph()
                self.device_output_graph.sequential(
                    [
                        self.samna_device.get_model().get_source_node(),
                        self.samna_output_buffer,
                    ]
                )

                self.device_input_graph.start()
                self.device_output_graph.start()
                self.samna_config = config

                return self

            else:
                self._to_device(device)

        else:
            raise Exception("Unknown device description.")

    def make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="speck2fdevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ):
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Args:
            chip_layers_ordering: sequence of integers or `auto`. The order in
                which the dynapcnn layers will be used. If `auto`, an automated
                procedure will be used to find a valid ordering. A list of
                layers on the device where you want each of the model's
                DynapcnnLayers to be placed. Note: This list should be the same
                length as the number of dynapcnn layers in your model.
            device (str): speck2edevkit or speck2fdevkit
            monitor_layers: A list of all layers in the module that you want to
                monitor. Indexing starts with the first non-dvs layer. If you
                want to monitor the dvs layer for eg.
                ::

                    monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                    monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                    monitor_layers = "all" # If you want to monitor all the layers

                If this value is left as None, by default the last layer of the model is monitored.
            config_modifier: A user configuration modifier method. This
                function can be used to make any custom changes you want to
                make to the configuration object.

        Returns:
            Object defining the configuration for the device.

        Raises:
            ImportError: If samna is not available.
            ValueError: If the generated configuration is not valid for the specified device.
        """
        config, is_compatible = self._make_config(
            chip_layers_ordering=chip_layers_ordering,
            device=device,
            monitor_layers=monitor_layers,
            config_modifier=config_modifier,
        )
        if is_compatible:
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")

    def is_compatible_with(self, device_type: str) -> bool:
        pass

    def make_config(
        self,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        device: str = "speck2fdevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ):
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Args:
            layer2core_map (dict or "auto"): Defines how cores on chip are
                assigned to DynapcnnLayers. If `auto`, an automated procedure
                will be used to find a valid ordering. Otherwise a dict needs
                to be passed, with DynapcnnLayer indices as keys and assigned
                core IDs as values. DynapcnnLayer indices have to match those of
                `self.dynapcnn_layers`.
            device: (string): speck2devkit
            monitor_layers: A list of all layers in the module that you want to
                monitor. Indexing starts with the first non-dvs layer. If you
                want to monitor the dvs layer for eg.
                ::

                    monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                    monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                    monitor_layers = "all" # If you want to monitor all the layers
                    monitor_layers = [-1] # If you want to only monitor exit points of the network (i.e. final layers)

                If this value is left as None, by default the last layer of the
                model is monitored.
            config_modifier (Callable or None): A user configuration modifier
                method. This function can be used to make any custom changes
                you want to make to the configuration object.
            chip_layers_ordering (None, sequence of integers or "auto", obsolete):
                The order in which the dynapcnn layers will be used. If `auto`,
                an automated procedure will be used to find a valid ordering.
                A list of layers on the device where you want each of the model's
                DynapcnnLayers to be placed. Note: This list should be the same
                length as the number of dynapcnn layers in your model. Note:
                This parameter is obsolete and should not be passed anymore.
                Use `layer2core_map` instead.

        Returns:
            Object defining the configuration for the device

        Raises:
            ImportError: If samna is not available.
            ValueError: If the generated configuration is not valid for the specified device.
        """
        config, is_compatible = self._make_config(
            layer2core_map=layer2core_map,
            device=device,
            monitor_layers=monitor_layers,
            config_modifier=config_modifier,
            chip_layers_ordering=chip_layers_ordering,
        )

        if is_compatible:
            print("Network is valid")
            return config
        else:
            raise ValueError(
                f"Generated config is not valid for {device}. "
                "Probably one or more layers are too large. Try "
                "Reducing the number of neurons or the kernel sizes."
            )

    def has_dvs_layer(self) -> bool:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def reset_states(self, randomize=False):
        """Reset the states of the network.
        Note that setting `randomize` to `True` is only supported for models
        that have not yet been deployed on a SynSense device.

        Args:
            randomize (bool): If `False` (default), will set all states to 0.
                Otherwise will set to random values.
        """
        if hasattr(self, "device") and isinstance(self.device, str):  # pragma: no cover
            device_name, _ = parse_device_id(self.device)
            if device_name in ChipFactory.supported_devices:
                config_builder = ChipFactory(self.device).get_config_builder()
                config_builder.reset_states(self.samna_config, randomize=randomize)
                self.samna_device.get_model().apply_configuration(self.samna_config)
                time.sleep(1)
                if not randomize:
                    if hasattr(self, "samna_input_graph"):
                        self.samna_input_graph.stop()
                        for lyr_idx in self.chip_layers_ordering:
                            config_builder.set_all_v_mem_to_zeros(
                                self.samna_device, lyr_idx
                            )
                            time.sleep(0.1)
                        self.samna_input_graph.start()
                return

        for layer in self.sequence:
            if isinstance(layer, DynapcnnLayer):
                layer.spk_layer.reset_states(randomize=randomize)

    def _make_config(
        self,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        device: str = "speck2fdevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ) -> Tuple["SamnaConfiguration", bool]:
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Args:
            layer2core_map (dict or "auto"): Defines how cores on chip are
                assigned to DynapcnnLayers. If `auto`, an automated procedure
                will be used to find a valid ordering. Otherwise a dict needs
                to be passed, with DynapcnnLayer indices as keys and assigned
                core IDs as values. DynapcnnLayer indices have to match those
                of `self.dynapcnn_layers`.
            device: (string): dynapcnndevkit, speck2b or speck2devkit
            monitor_layers: A list of all layers in the module that you want
                to monitor. Indexing starts with the first non-dvs layer.
                If you want to monitor the dvs layer for eg.
                ::

                    monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                    monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                    monitor_layers = "all" # If you want to monitor all the layers
                    monitor_layers = [-1] # If you want to only monitor exit points of the network (i.e. final layers)

                If this value is left as None, by default the last layer of the model is monitored.

            config_modifier (Callable or None): A user configuration modifier
                method. This function can be used to make any custom changes
                you want to make to the configuration object.
            chip_layers_ordering (None, sequence of integers or "auto", obsolete):
                The order in which the dynapcnn layers will be used. If `auto`,
                an automated procedure will be used to find a valid ordering.
                A list of layers on the device where you want each of the
                model's DynapcnnLayers to be placed. Note: This list should be
                the same length as the number of dynapcnn layers in your model.
                Note: This parameter is obsolete and should not be passed
                anymore. Use `layer2core_map` instead.

        Returns:
            An object defining the configuration for the device and a boolean
            that determines if the configuration is valid for the given device.

        Raises:
            ImportError: If samna is not available.
            ValueError: If no valid mapping between the layers of this object
                and the cores ofthe provided device can be found.
        """
        config_builder = ChipFactory(device).get_config_builder()

        if chip_layers_ordering is not None:
            if layer2core_map != "auto":
                warn(
                    "Both `chip_layers_ordering` and `layer2core_map are provided. "
                    "The parameter `chip_layers_ordering` is deprecated and will "
                    "be ignored.",
                    DeprecationWarning,
                )
            elif chip_layers_ordering == "auto":
                warn(
                    "The parameter `chip_layers_ordering` is deprecated. Passing "
                    "'auto' is still accepted, but in the future please use "
                    "`layer2core_map` instead.",
                    DeprecationWarning,
                )
            else:
                layer2core_map = {
                    idx: core
                    for idx, core in zip(self.dynapcnn_layers, chip_layers_ordering)
                }
                warn(
                    "The parameter `chip_layers_ordering` is deprecated. "
                    "Because `layer2core_map` is 'auto', and `chip_layers_ordering` "
                    "is not, will convert `chip_layers_ordering` to a "
                    "dict matching `layer2core_map`. In the future please use "
                    "`layer2core_map` instead. Please make sure the inferred "
                    "mapping from DynapcnnLayer index to core index is correct: "
                    + pformat(layer2core_map),
                    DeprecationWarning,
                )
        if layer2core_map == "auto":
            layer2core_map = config_builder.map_layers_to_cores(self.dynapcnn_layers)
        else:
            if not layer2core_map.keys() == self.dynapcnn_layers.keys():
                raise ValueError(
                    "The keys provided in `layer2core_map` must exactly match "
                    "the keys in `self.dynapcnn_layers`"
                )

        self._layer2core_map = layer2core_map

        config = config_builder.build_config(
            layers=self.all_layers,
            layer2core_map=layer2core_map,
            destination_map=self.layer_destination_map,
        )

        if monitor_layers is None:
            monitor_layers = self._dynapcnn_module.get_exit_layers()
        elif monitor_layers == "all":
            monitor_layers = [
                lyr_idx
                for lyr_idx, layer in self.dynapcnn_layers.items()
                if not isinstance(layer, DVSLayer)
            ]
        elif -1 in monitor_layers:
            monitor_layers.remove(-1)
            monitor_layers += self._dynapcnn_module.get_exit_layers()

        monitor_chip_layers = []
        for lyr_idx in monitor_layers:
            if str(lyr_idx).lower() == "dvs":
                monitor_chip_layers.append("dvs")
            else:
                monitor_chip_layers.append(layer2core_map[lyr_idx])

        config_builder.monitor_layers(config, monitor_chip_layers)

        if config_modifier is not None:
            config = config_modifier(config)

        return config, config_builder.validate_configuration(config)

    def _to_device(self, device: torch.device) -> None:
        """Access each sub-layer within all `DynapcnnLayer` instances and call `.to(device)` on them."""
        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                layer.to(device)

        for _, data in self._merge_points.items():
            data["merge"].to(device)

    def __str__(self):
        pretty_print = ""
        if self.dvs_layer is not None:
            pretty_print += (
                "-------------------------- [ DVSLayer ] --------------------------\n"
            )
            pretty_print += f"{self.dvs_layer}\n\n"
        for idx, layer_data in self.dynapcnn_layers.items():
            pretty_print += f"----------------------- [ DynapcnnLayer {idx} ] -----------------------\n"
            if self.is_deployed_on_dynapcnn_device:
                pretty_print += f"Core {self.layer2core_map[idx]}\n"
            pretty_print += f"{layer_data}\n\n"

        return pretty_print

    def __repr__(self):
        if self.is_deployed_on_dynapcnn_device:
            layer_info = "\n\n".join(
                f"{idx} - core: {self.layer2core_map[idx]}\n{pformat(layer)}"
                for idx, layer in self.dynapcnn_layers.items()
            )
            device_info = f" deployed on {self.device},"
        else:
            layer_info = "\n\n".join(
                f"Index: {idx}\n{pformat(layer)}"
                for idx, layer in self.dynapcnn_layers.items()
            )
            device_info = f" on {self.device}," if hasattr(self, "device") else ""
        return (
            f"DynapCNN Network{device_info} containing:\nDVS Layer: {pformat(self.dvs_layer)}"
            "\n\nDynapCNN Layers:\n\n" + layer_info
        )


class DynapcnnCompatibleNetwork(DynapcnnNetwork):

    def __init__(self, *args, **kwargs):
        from warnings import warn

        warn(
            "DynapcnnCompatibleNetwork has been renamed to DynapcnnNetwork "
            + "and will be removed in a future release."
        )
        super().__init__(*args, **kwargs)
