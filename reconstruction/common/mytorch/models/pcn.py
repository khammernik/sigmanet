"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch import nn
from common.mytorch.complex import ParallelComplexInstanceNorm
from common.mytorch.models.datalayer import DCLayer


class ParallelCoilComplexNormWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.complex_instance_norm = ParallelComplexInstanceNorm(15)

    def forward(self, input, attrs):

        # compute complex instance norm on sample
        # of size [nBatch, nCoils, nFE, nPE, 2]
        self.complex_instance_norm.set_normalization(
            attrs['mean'],
            attrs['cov'],
        )
        output = self.complex_instance_norm.normalize(input)

        # re-shape data from [nBatch, nCoils, nFE, nPE, 2]
        # to [nBatch, nCoils * 2, nFE, nPE]
        shp = output.shape
        output = output.permute(0, 1, 4, 2, 3).contiguous().view(
            shp[0], shp[1] * shp[4], shp[2], shp[3])

        # apply denoising
        output = self.model(output)

        # re-shape data from [nBatch, nCoils * 2, nFE, nPE]
        # to [nBatch, nCoils, nFE, nPE, 2]
        output = output.view(shp[0], shp[1], shp[4], shp[2], shp[3]).permute(
            0, 1, 3, 4, 2)
        # unnormalize
        output = self.complex_instance_norm.unnormalize(output)
        return output


class ParallelCoilDCWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, mask):
        x_dc = self.model(x.unsqueeze(2), y, mask)
        return x_dc.squeeze(2)


class ParallelCoilNetwork(torch.nn.Module):
    """
    Parallel coil network with data term based on forward and adjoint
    performed for each coil

    """
    def __init__(
        self,
        num_iter,
        model,
        model_config,
        datalayer_config,
        save_space=False,
        reset_cache=False,
        shared_params=True,
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
            ParallelCoilComplexNormWrapper(model(**model_config))
            for i in range(self.num_iter)
        ])
        self.gradD = torch.nn.ModuleList([
            ParallelCoilDCWrapper(DCLayer(**datalayer_config))
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
            x = self.gradD[i%self.num_iter](x_thalf, y, mask)
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
            x = self.gradD[i%self.num_iter](x_thalf, y, mask)

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
