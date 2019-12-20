"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch import nn
from common.mytorch.complex import ComplexInstanceNorm


class ComplexNormWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.complex_instance_norm = ComplexInstanceNorm()

    def forward(self, input, attrs):
        # compute complex instance norm on sample
        # of size [nBatch, nSmaps, nFE, nPE, 2]
        self.complex_instance_norm.set_normalization(
            attrs['mean'],
            attrs['cov'],
        )
        output = self.complex_instance_norm.normalize(input)

        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 2, nFE, nPE]
        shp = output.shape
        output = output.view(shp[0] * shp[1], *shp[2:]).permute(0, 3, 1, 2)

        # apply denoising
        output = self.model(output)

        # re-shape data from [nBatch*nSmaps, 2, nFE, nPE]
        # to [nBatch, nSmaps, nFE, nPE, 2]
        output = output.permute(0, 2, 3, 1).view(*shp)
        # unnormalize
        output = self.complex_instance_norm.unnormalize(output)
        return output


class SensitivityNetwork(torch.nn.Module):
    """
    Sensitivity network with data term based on forward and adjoint containing
    the sensitivity maps

    """
    def __init__(
        self,
        num_iter,
        model,
        model_config,
        datalayer,
        datalayer_config,
        shared_params=True,
        save_space=False,
        reset_cache=False,
    ):
        super().__init__()

        self.shared_params = shared_params

        if self.shared_params:
            self.num_iter = 1
        else:
            self.num_iter = num_iter

        self.num_iter_total = num_iter

        self.is_trainable = [True] * num_iter

        # setup the modules
        self.gradR = torch.nn.ModuleList([
            ComplexNormWrapper(model(**model_config))
            for i in range(self.num_iter)
        ])
        self.gradD = torch.nn.ModuleList([
            datalayer(**datalayer_config)
            for i in range(self.num_iter)
        ])

        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space
        self.reset_cache = reset_cache

    def forward(self, x, y, smaps, mask, attrs):
        x_all = [x]
        x_half_all = []
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)

        for i in range(num_iter):
            x_thalf = x - self.gradR[i%self.num_iter](x, attrs)
            x = self.gradD[i%self.num_iter](x_thalf, y, smaps, mask)
            x_all.append(x)
            x_half_all.append(x_thalf)

        return x_all[-1]

    def forward_save_space(self, x, y, smaps, mask, attrs):
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)
        
        for i in range(num_iter):
            x_thalf = x - self.gradR[i%self.num_iter](x, attrs)
            x = self.gradD[i%self.num_iter](x_thalf, y, smaps, mask)

            # would run out of memory at test time
            # if this is False for some cases
            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()

        return x

    def freeze(self, i):
        """ freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = False
        self.is_trainable[i] = False

    def unfreeze(self, i):
        """ freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = True
        self.is_trainable[i] = True

    def freeze_all(self):
        """ freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.freeze(i)

    def unfreeze_all(self):
        """ freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.unfreeze(i)

    def copy_params(self, src_i, trg_j):
        """ copy i-th cascade net parameters to j-th cascade net parameters """
        src_params = self.gradR[src_i].parameters()
        trg_params = self.gradR[trg_j].parameters()

        for i, (trg_param, src_param) in enumerate(
                zip(trg_params, src_params)):
            trg_param.data.copy_(src_param.data)

    def stage_training_init(self):
        self.freeze_all()
        self.unfreeze(0)
        print(self.is_trainable)

    def stage_training_transition_i(self, copy=False):
        if not self.shared_params:
            # if all unlocked, don't do anything
            if not np.all(self.is_trainable):
                for i in range(self.num_iter):

                    # if last cascade is reached, unlock all
                    if i == self.num_iter - 1:
                        self.unfreeze_all()
                        break

                    # freeze current i, unlock next. copy parameter if specified
                    if self.is_trainable[i]:
                        self.freeze(i)
                        self.unfreeze(i+1)
                        if copy:
                            self.copy_params(i, i+1)
                        break