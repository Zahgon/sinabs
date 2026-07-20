
from typing import Union

import torch.nn as nn

import sinabs.layers as sl

from .dvs_layer import DVSLayer

Pooling = (sl.SumPool2d, nn.AvgPool2d)
Weight = (nn.Conv2d, nn.Linear)
Neuron = (sl.IAFSqueeze,)
DVS = (DVSLayer,)
SupportedNodeTypes = (*Pooling, *Weight, *Neuron, *DVS)

VALID_SINABS_EDGE_TYPES_ABSTRACT = {
    (Weight, Neuron): "weight-neuron",
    (Neuron, Pooling): "neuron-pooling",
    (Pooling, Pooling): "pooling-pooling",
    (Neuron, Weight): "neuron-weight",
    (Pooling, Weight): "pooling-weight",
    (DVS, Weight): "dvs-weight",
    (DVS, Pooling): "dvs-pooling",
}

VALID_SINABS_EDGE_TYPES = {
    (source_type, target_type): name
    for types, name in VALID_SINABS_EDGE_TYPES_ABSTRACT.items()
    for source_type in types[0]
    for target_type in types[1]
}

LAYER_TYPES_WITH_MULTIPLE_INPUTS = (sl.Merge,)

LAYER_TYPES_WITH_MULTIPLE_OUTPUTS = (*Neuron, *Pooling, *DVS)
