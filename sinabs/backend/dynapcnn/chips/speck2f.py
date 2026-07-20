from typing import Dict, List

import samna
from samna.speck2f.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer

from .dynapcnn import DynapcnnConfigBuilder


class Speck2FConfigBuilder(DynapcnnConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.speck2f

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_dvs_layer_config(cls):
        pass

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2f_event_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2f_event_output_event()
