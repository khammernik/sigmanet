"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import common.mytorch as mytorch
from common.mytorch.fft import fft2, ifft2
from common.mytorch.mri import (
    adjointSoftSenseOpNoShift,
    forwardSoftSenseOpNoShift,
)


class DataIDLayer(nn.Module):
    """
        Placeholder for data layer
    """
    def __init__(self, *args, **kwargs):
        super(DataIDLayer, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x

    def __repr__(self):
        return f'DataIDLayer()'


class DataGDLayer(nn.Module):
    """
        DataLayer computing the gradient on the L2 dataterm.
    """
    def __init__(self, lambda_init, learnable=True):
        """
        Args:
            lambda_init (float): Init value of data term weight lambda.
        """
        super(DataGDLayer, self).__init__()
        self.lambda_init = lambda_init
        self.data_weight = torch.nn.Parameter(torch.Tensor(1))
        self.data_weight.data = torch.tensor(
            lambda_init,
            dtype=self.data_weight.dtype,
        )
        self.data_weight.requires_grad = learnable

    def forward(self, x, y, smaps, mask):
        A_x_y = forwardSoftSenseOpNoShift(x, smaps, mask) - y
        gradD_x = adjointSoftSenseOpNoShift(A_x_y, smaps, mask)
        return x - self.data_weight * gradD_x

    def __repr__(self):
        return f'DataLayer(lambda_init={self.data_weight.item():.4g})'


class DataProxCGLayer(torch.nn.Module):
    """ Solving the prox wrt. dataterm using Conjugate Gradient as proposed by
    Aggarwal et al.
    """
    def __init__(self, lambda_init, tol=1e-6, iter=10, learnable=True):
        super(DataProxCGLayer, self).__init__()

        self.lambdaa = torch.nn.Parameter(torch.Tensor(1))
        self.lambdaa.data = torch.tensor(lambda_init)
        self.lambdaa_init = lambda_init
        self.lambdaa.requires_grad = learnable

        self.tol = tol
        self.iter = iter

        self.op = MyCG

    def forward(self, x, f, smaps, mask):
        return self.op.apply(
            x,
            self.lambdaa,
            f,
            smaps,
            mask,
            self.tol,
            self.iter,
        )

    def extra_repr(self):
        return (f"lambda_init={self.lambdaa.item():.4g}, tol={self.tol}"
                f" iter={self.iter} learnable={self.lambdaa.requires_grad}")

    def set_learnable(self, flag):
        self.lambdaa.requires_grad = flag


class MyCG(torch.autograd.Function):
    @staticmethod
    def complexDot(data1, data2):
        nBatch = data1.shape[0]
        mult = mytorch.complex.complex_mult_conj(data1, data2)
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re.view(nBatch, -1), dim=-1),
                            torch.sum(im.view(nBatch, -1), dim=-1)], -1)

    @staticmethod
    def solve(x0, M, tol, max_iter):
        nBatch = x0.shape[0]
        x = torch.zeros(x0.shape).to(x0.device)
        r = x0.clone()
        p = x0.clone()
        x0x0 = (x0.pow(2)).view(nBatch, -1).sum(-1)
        rr = torch.stack([
            (r.pow(2)).view(nBatch, -1).sum(-1),
            torch.zeros(nBatch).to(x0.device)
        ], dim=-1)

        it = 0
        while torch.min(rr[..., 0] / x0x0) > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = mytorch.complex.complex_div(rr, MyCG.complexDot(p, q))
            # alpha = torch.stack([rr[...,0] / MyCG.complexDot(p, q)[...,0],
            #                      torch.zeros(nBatch).to(x0.device)], dim=-1)
            x += mytorch.complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1), p.clone())

            r -= mytorch.complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1), q.clone())

            rr_new = torch.stack([
                (r.pow(2)).view(nBatch, -1).sum(-1),
                torch.zeros(nBatch).to(x0.device)
            ], dim=-1)

            beta = torch.stack([
                rr_new[..., 0] / rr[..., 0],
                torch.zeros(nBatch).to(x0.device)
            ], dim=-1)

            p = r.clone() + mytorch.complex.complex_mult(
                beta.reshape(nBatch, 1, 1, 1, -1), p)

            rr = rr_new.clone()
            # print(it, rr[...,0]/x0x0)
        return x

    @staticmethod
    def forward(ctx, z, lambdaa, y, smaps, mask, tol, max_iter):
        ctx.tol = tol
        ctx.max_iter = max_iter

        def A(x):
            return mytorch.mri.forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return mytorch.mri.adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return lambdaa * AT(A(p)) + p

        x0 = lambdaa * AT(y) + z
        ctx.save_for_backward(AT(y), x0, smaps, mask, lambdaa)

        return MyCG.solve(x0, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        ATy, rhs, smaps, mask, lambdaa = ctx.saved_tensors

        def A(x):
            return mytorch.mri.forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return mytorch.mri.adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return lambdaa * AT(A(p)) + p

        Qe  = MyCG.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = MyCG.solve(Qe,     M, ctx.tol, ctx.max_iter)

        grad_z = Qe

        grad_lambdaa = mytorch.complex.complex_dotp(Qe,  ATy).sum() \
                     - mytorch.complex.complex_dotp(QQe, rhs).sum()

        return grad_z, grad_lambdaa, None, None, None, None, None


class DataVSLayer(nn.Module):
    """
        DataLayer using variable splitting formulation
    """
    def __init__(self, alpha_init, beta_init, learnable=True):
        """
        Args:
            alpha_init (float): Init value of data consistency block (DCB)
            beta_init (float): Init value of weighted averaging block (WAB)
        """
        super(DataVSLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.alpha.data = torch.tensor(alpha_init, dtype=self.alpha.dtype)

        self.beta = torch.nn.Parameter(torch.Tensor(1))
        self.beta.data = torch.tensor(beta_init, dtype=self.beta.dtype)

        self.learnable = learnable
        self.set_learnable(learnable)

    def forward(self, x, y, smaps, mask):
        A_x = mytorch.mri.forwardSoftSenseOpNoShift(x, smaps, 1.)
        k_dc = (1 - mask) * A_x + mask * (
            self.alpha * A_x + (1 - self.alpha) * y)
        x_dc = mytorch.mri.adjointSoftSenseOpNoShift(k_dc, smaps, 1.)
        x_wab = self.beta * x + (1 - self.beta) * x_dc
        return x_wab

    def extra_repr(self):
        return (
            f"alpha={self.alpha.item():.4g},"
            f"beta={self.beta.item():.4g},"
            f"learnable={self.learnable}"
        )

    def set_learnable(self, flag):
        self.learnable = flag
        self.alpha.requires_grad = self.learnable
        self.beta.requires_grad = self.learnable


class DCLayer(nn.Module):
    """
        Data Consistency layer from DC-CNN, apply for single coil mainly
    """
    def __init__(self, lambda_init=0., learnable=True):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(DCLayer, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.Tensor(1))
        self.lambda_.data = torch.tensor(lambda_init, dtype=self.lambda_.dtype)

        self.learnable = learnable
        self.set_learnable(learnable)

    def forward(self, x, y, mask):
        A_x = fft2(x)
        k_dc = (1 - mask) * A_x + mask * (
            self.lambda_ * A_x + (1 - self.lambda_) * y)
        x_dc = ifft2(k_dc)
        return x_dc

    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4g},learnable={self.learnable}"

    def set_learnable(self, flag):
        self.learnable = flag
        self.lambda_.requires_grad = self.learnable
