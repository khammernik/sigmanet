"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F
from common.mytorch.tensor_ops import center_crop


def calculate_downsampling_padding2d(tensor, num_pool_layers):
    # calculate pad size
    factor = 2 ** num_pool_layers
    imshape = np.array(tensor.shape[-2:])
    paddings = np.ceil(imshape / factor) * factor- imshape
    paddings = paddings.astype(np.int) // 2
    p2d = (paddings[1], paddings[1], paddings[0], paddings[0])
    return p2d

def pad2d(tensor, p2d):
    if np.any(p2d):
        # order of padding is reversed. that's messed up.
        tensor = F.pad(tensor, p2d)
    return tensor

def unpad2d(tensor, shape):
    if tensor.shape == shape:
        return tensor
    else:
        return center_crop(tensor, shape)


class InstanceNormWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        """
        Args:
          input: (nb, nc, nx, ny)
        Returns:
          output: (nb, nc, nx, ny)
        """
        # compute stats
        mean = input.mean(dim=(-1, -2), keepdim=True)
        std = input.std(dim=(-1, -2), keepdim=True)

        # normalize
        normalized_input = (input - mean) / (std + 1e-9)

        # remove extreme values
        normalized_input = normalized_input.clamp(-6, 6)

        # network
        output = self.model(normalized_input)

        # unnormalize
        output = output * std + mean

        return output
