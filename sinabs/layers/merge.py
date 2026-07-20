import torch.nn as nn


class Merge(nn.Module):
    def __init__(self) -> None:
        """Module form for a merge operation.

        In the context of events/spikes, events/spikes from two different sources/rasters will be
        added.
        """
        super().__init__()

    def forward(self, data1, data2):
        pass
