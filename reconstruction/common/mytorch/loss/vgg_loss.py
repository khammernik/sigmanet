"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from . import modified_vgg


class VGGLoss(torch.nn.Module):
    def __init__(
            self,
            layer_name_mapping=None,
            normalize=True,
            device='gpu',
            vgg_model=None,
            full=False,
            inplace=False,
            distance=2,
    ):
        super(VGGLoss, self).__init__()
        self.layer_name_mapping = layer_name_mapping
        if self.layer_name_mapping is None:
            self.layer_name_mapping = {
                '0': 'conv1_0',
                # '1': 'relu1_0',
                '2': "conv1_1",
                # '3': 'relu1_1',
                '7': "conv2_2",
                # '8': "relu2_2",
                '14': "conv3_3",
                # '15': "relu3_3",
                '21': "conv4_3",
                # '22': "relu4_3",  # <- gradient is strangely huge... turn off for now
            }

        self.normalize = normalize
        self.device = device
        self.full = full
        if distance == 1:
            self.distance = F.l1_loss
        else:
            self.distance = F.mse_loss

        if vgg_model is None:
            if inplace:
                vgg_model = vgg.vgg16(pretrained=True)
            else:
                vgg_model = modified_vgg.vgg16(pretrained=True)

        vgg_model.to(self.device)
        vgg_model.eval()

        self.vgg_layers = vgg_model.features
        del vgg_model

        # normalizatoin
        self.mean_t = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.std_t = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.mean_t = self.mean_t.view(1, 3, 1, 1).to(self.device)
        self.std_t = self.std_t.view(1, 3, 1, 1).to(self.device)

    def _normalize(self, x, max_val=None):
        # scale image to [0, 1] range, then perform color whitening
        if max_val is None:
            max_val = x.max()
        return ((x/max_val) - self.mean_t) / self.std_t

    def forward(self, target, pred, max_val=None):
        num_chans = target.shape[1]
        assert num_chans in [1, 3]
        assert target.shape == pred.shape

        if num_chans == 1:
            target = torch.cat([target,target,target], 1)
            pred = torch.cat([pred,pred,pred], 1)

        if self.normalize:
            target = self._normalize(target, max_val)
            pred = self._normalize(pred, max_val)

        ctr = 0
        total_loss = 0
        output = {}
        losses = {}
        n_layers = float(len(self.layer_name_mapping))
        for name, module in self.vgg_layers._modules.items():
            target = module(target)
            pred = module(pred)

            if name in self.layer_name_mapping:
                curr_loss = self.distance(pred, target)
                total_loss += curr_loss
                if self.full:
                    losses[self.layer_name_mapping[name]] = curr_loss
                    output[self.layer_name_mapping[name]] = (pred, target)
                ctr += 1

            # stop early to save some space
            if ctr == len(self.layer_name_mapping):
                break

        if self.full:
            return total_loss / n_layers, losses, output
        else:
            return total_loss / n_layers
