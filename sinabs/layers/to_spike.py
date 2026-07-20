from typing import Tuple

import torch
from torch import nn


class Img2SpikeLayer(nn.Module):

    def __init__(
        self,
        image_shape,
        tw: int = 100,
        max_rate: float = 1000,
        norm: float = 255.0,
        squeeze: bool = False,
        negative_spikes: bool = False,
    ):
        super().__init__()
        self.tw = tw
        self.max_rate = max_rate
        self.norm = norm
        self.squeeze = squeeze
        self.negative_spikes = negative_spikes

    def forward(self, img_input):
        pass

    def get_output_shape(self, input_shape: Tuple):
        pass


class Sig2SpikeLayer(torch.nn.Module):

    def __init__(
        self,
        channels_in,
        tw: int = 1,
        norm_level: float = 1,
        spk_out: bool = True,
    ):
        super().__init__()
        self.tw = tw
        self.norm_level = norm_level
        self.spk_out = spk_out

    def get_output_shape(self, input_shape: Tuple):
        pass

    def forward(self, signal):
        pass
