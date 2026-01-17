import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if isinstance(input, np.ndarray):
        if input.dtype == np.object_:
            input = input.astype(np.float32)  # Convert to a supported type
        return torch.from_numpy(input)
    return input
