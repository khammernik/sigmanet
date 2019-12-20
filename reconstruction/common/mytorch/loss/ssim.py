"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian_window1d(win_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - win_size // 2) ** 2 / (2.0 * sigma ** 2))
        for x in range(win_size)
    ])
    return gauss / gauss.sum()


def uniform_window1d(win_size=7):
    return torch.ones(win_size, dtype=torch.float32) / win_size


def create_window(win_size, channel=1, gaussian_weights=False):
    if gaussian_weights:
        _1D_window = gaussian_window1d(win_size, 1.5).unsqueeze(1)
    else:
        _1D_window = uniform_window1d(win_size).unsqueeze(1)

    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, win_size, win_size).contiguous()
    return window


def ssim(
        gt,
        pred,
        window=None,
        win_size=None,
        gaussian_weights=False,
        full=False,
        data_range=None,
        size_average=True,
        **kwargs,
):
    """ Pytorch version of ndimage.metrics.structural_similarity except for the
    following differences:

       - can pass the window kernel via `window`
       - currently only supports: float32, 2d, `multichannel=True`
       - `multichnnel` is always True along the channel dim of the tensor
       - always computed in batch (with size_average flag)
       - `data_range` can be an array
    """

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if data_range is None:
        # in skimage, this is 0.0-1.0 for float. We overwrite to be the maximum
        # of the target
        data_range = gt.max()

    # check the data_range dimension
    if not isinstance(data_range, torch.Tensor):
        data_range = torch.Tensor(data_range).to(gt.device)
    if data_range.dim() < gt.dim():
        # curently only supports 2d anyways
        data_range = data_range.view(-1, 1, 1, 1)

    (batchsize, channel, height, width) = gt.size()
    if win_size is None:
        if window is not None:
            win_size = window.shape[-1]
        elif gaussian_weights:
            sigma = kwargs.pop('sigma', 1.5)
            truncate = 3.5
            r = int(truncate * sigma + 0.5)
            win_size = 2 * r + 1
        else:
            win_size = 7

    if window is None:
        win_size = min(win_size, height, width)
        window = create_window(win_size, channel, gaussian_weights).to(gt.device)

    NP = win_size ** 2
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)
    else:
        cov_norm = 1.0

    # compute padding
    pad = win_size // 2

    # compute (weighted) means
    ux = F.conv2d(gt, window, groups=channel, padding=pad)
    uy = F.conv2d(pred, window, groups=channel, padding=pad)

    ux_sq = ux.pow(2)
    uy_sq = uy.pow(2)
    ux_uy = ux * uy

    # compute (weighted) variances and covariances
    uxx = F.conv2d(gt * gt, window, groups=channel, padding=pad) - ux_sq
    uyy = F.conv2d(pred * pred, window, groups=channel, padding=pad) - uy_sq
    uxy = F.conv2d(gt * pred, window, groups=channel, padding=pad) - ux_uy
    vx = cov_norm * uxx
    vy = cov_norm * uyy
    vxy = cov_norm * uxy

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    S = ((2.0 * ux * uy + C1) * (2.0 * vxy + C2)) / ((ux_sq + uy_sq + C1) * (vx + vy + C2))

    # to avoid edge effects will ignore filter radius strip around edges
    ssim_map = S[:, :, pad:-pad, pad:-pad]

    # compute (weighted) mean of ssim
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.view(batchsize, -1).mean(-1)

    if full:
        return ret, S
    else:
        return ret


def msssim(gt, pred, win_size=11, size_average=True, data_range=None, normalize=False):
    device = gt.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(gt, pred, win_size=win_size, size_average=size_average, full=True, data_range=data_range)
        mssim.append(sim)
        mcs.append(cs)

        gt = F.avg_pool2d(gt, (2, 2))
        pred = F.avg_pool2d(pred, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(
            self,
            win_size=7,
            gaussian_weights=False,
            channel=1,
            size_average=True,
            device='cuda',
            **kwargs,
    ):
        super(SSIM, self).__init__()
        self.device = device
        self.win_size = win_size
        self.channel = channel
        self.gaussian_weights = gaussian_weights
        self.size_average = size_average
        self.window = create_window(
            self.win_size,
            self.channel,
            self.gaussian_weights,
        ).to(self.device)
        self.kwargs = kwargs

    def forward(self, gt, pred, full=False, data_range=None):
        return ssim(
            gt,
            pred,
            window=self.window,
            win_size=self.win_size,
            full=full,
            data_range=data_range,
            size_average=self.size_average,
            **self.kwargs,
        )

class MSSSIM(torch.nn.Module):
    def __init__(self, win_size=11, size_average=True, channel=3, device='cuda'):
        super(MSSSIM, self).__init__()
        self.win_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.device = device

    def forward(self, gt, pred):
        return msssim(
            gt,
            pred,
            win_size=self.win_size,
            size_average=self.size_average,
            device=self.device,
        )
