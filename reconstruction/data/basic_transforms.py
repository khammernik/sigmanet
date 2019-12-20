"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from common.mytorch.complex import complex_abs
from common.mytorch.tensor_ops import center_crop


class Magnitude(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        # Crop input image
        for k in self.keys:
            sample[k] = complex_abs(sample[k], eps=1e-9)
        return sample


class Crop(object):
    def __init__(self, resolution, keys):
        self.resolution = resolution
        self.keys = keys

    def __call__(self, sample):
        # Crop input image
        for k in self.keys:
            x = sample[k]
            nx, ny = x.shape[-2], x.shape[-1]
            sample[k] = center_crop(x, (min(nx, self.resolution), min(ny, self.resolution)))
        return sample
