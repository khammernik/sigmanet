"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

def weight_init(module):
    if isinstance(module, torch.nn.Conv2d) \
       or isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            module.weight, a=0,
            mode='fan_in',
            nonlinearity='relu',
        )
        if module.bias is not None:
            module.bias.data.fill_(0)
