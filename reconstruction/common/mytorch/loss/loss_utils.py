"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradientNormalizer(torch.autograd.Function):
    """ Normalise the gradient for given input.

    Usage: suppose we have

    L(y_gt, x), we get dL/dx.



    which will give gradient dL/dy to gradient input to function f, such that

    dL/dx = dL/dy * dy/dx

    By using this function as:

    L(y_gt, y), y = f(GradientNormalizer(wt)(x))

    we get: (dL/dx)_new = wt * dL/dx / ||dL/dx||.

    Maybe useful for scaling multiple losses

    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(ctx, input):
        return input

    def backward(ctx, grad_output):
        norm = grad_output.norm()
        # print('gradient norm', norm, norm / ctx.weight)
        # print(grad_output[0,0,:5,:5])
        norm_grad = grad_output / (norm / ctx.weight)
        # print('norm gradient norm', norm_grad.norm())
        # print(norm_grad[0,0,:5,:5])
        # print('gradient norm', norm)
        #        norm_grad = grad_output / norm * ctx.weight
        # print('norm gradient norm', norm_grad.norm())
        return norm_grad


class GradientNormalizedLoss(torch.nn.Module):
    """ Normalise the gradient for given input.

    Usage: suppose we have

    L(y_gt, x), we get dL/dx.



    which will give gradient dL/dy to gradient input to function f, such that

    dL/dx = dL/dy * dy/dx

    By using this function as:

    L(y_gt, y), y = f(GradientNormalizer(wt)(x))

    we get: (dL/dx)_new = wt * dL/dx / ||dL/dx||.

    Maybe useful for scaling multiple losses

    """

    def __init__(self, loss_fn, weight=1.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.grad_normalize = GradientNormalizer(weight)

    def forward(self, target, pred):
        return self.loss_fn(target, self.grad_normalize(pred))
