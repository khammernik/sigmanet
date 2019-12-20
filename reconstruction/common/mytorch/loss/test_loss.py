"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
import torch

import common.mytorch.loss as loss

from common import utils
from common.evaluate import mse, nmse, psnr, ssim


def create_input_pair(shape, sigma=0.1):
    input = np.arange(np.product(shape)).reshape(shape).astype(float)
    input /= np.max(input)
    input2 = input + np.random.normal(0, sigma, input.shape)
    input = torch.from_numpy(input).float()
    input2 = torch.from_numpy(input2).float()
    return input, input2


@pytest.mark.parametrize('shape', [
    [10, 32, 32],
    [10, 64, 64],
])
def test_psnr(shape):
    input, input2 = create_input_pair(shape, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.psnr(input, input2, batch=False).item()
    err_numpy = psnr(input_numpy, input2_numpy)
    assert np.allclose(err, err_numpy)


def test_psnr_batch():
    shape4d = [4, 6, 32, 32]
    input, input2 = create_input_pair(shape4d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.psnr(input, input2, batch=True).item()
    err_numpy = 0
    for i in range(shape4d[0]):
        err_curr = psnr(input_numpy[i], input2_numpy[i])
        err_numpy += err_curr
    err_numpy /= shape4d[0]
    assert np.allclose(err, err_numpy)

    shape5d = [4, 6, 1, 32, 32]
    input, input2 = create_input_pair(shape5d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.psnr(input, input2)
    err_numpy = 0
    for i in range(shape5d[0]):
        err_numpy += psnr(input_numpy[i][:,0], input2_numpy[i][:,0])
    err_numpy /= shape5d[0]

    assert np.allclose(err, err_numpy)


@pytest.mark.parametrize('shape', [
    [5, 320, 320],
    [10, 64, 64],
])
def test_ssim(shape):
    input, input2 = create_input_pair(shape, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    torch_ssim = loss.SSIM(win_size=7, device='cpu')

    err = torch_ssim(input.unsqueeze(1), input2.unsqueeze(1)).item()
    err_numpy = ssim(input_numpy, input2_numpy)
    assert abs(err - err_numpy) < 1e-4


def test_ssim_batch():
    shape4d = [4, 6, 96, 96]
    input, input2 = create_input_pair(shape4d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    torch_ssim = loss.SSIM(win_size=7, device='cpu')

    data_range = input.view(4, -1).max(1)[0].repeat(6)
    err = torch_ssim(
        input.reshape(24, 1, 96, 96),
        input2.reshape(24, 1, 96, 96),
        data_range=data_range,
    ).item()
    err_numpy = 0
    for i in range(shape4d[0]):
        err_curr = ssim(input_numpy[i], input2_numpy[i])
        err_numpy += err_curr
    err_numpy /= shape4d[0]
    assert abs(err - err_numpy) < 1e-4


@pytest.mark.parametrize('shape', [
    [10, 32, 32],
    [10, 64, 64],
    [4, 6, 32, 32],
    [4, 6, 1, 32, 32],
])
def test_mse(shape):
    input, input2 = create_input_pair(shape, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = torch.nn.functional.mse_loss(input, input2).item()
    err_numpy = mse(input_numpy, input2_numpy)
    assert np.allclose(err, err_numpy)


def test_mse_batch():
    shape4d = [4, 6, 32, 32]
    input, input2 = create_input_pair(shape4d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = torch.nn.functional.mse_loss(input, input2).item()
    err_numpy = 0
    for i in range(shape4d[0]):
        err_curr = mse(input_numpy[i], input2_numpy[i])
        err_numpy += err_curr
    err_numpy /= shape4d[0]
    assert np.allclose(err, err_numpy)

    shape5d = [4, 6, 1, 32, 32]
    input, input2 = create_input_pair(shape5d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = torch.nn.functional.mse_loss(input, input2).item()
    err_numpy = 0
    for i in range(shape5d[0]):
        err_numpy += mse(input_numpy[i][:,0], input2_numpy[i][:,0])
    err_numpy /= shape5d[0]

    assert np.allclose(err, err_numpy)


@pytest.mark.parametrize('shape', [
    [10, 32, 32],
    [10, 64, 64],
    [4, 6, 32, 32],
    [4, 6, 1, 32, 32],
])
def test_nmse(shape):
    input, input2 = create_input_pair(shape, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.nmse(input, input2, batch=False).item()
    err_numpy = nmse(input_numpy, input2_numpy)
    assert np.allclose(err, err_numpy)


def test_nmse_batch():
    shape4d = [4, 6, 32, 32]
    input, input2 = create_input_pair(shape4d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.nmse(input, input2).item()
    err_numpy = 0
    for i in range(shape4d[0]):
        err_curr = nmse(input_numpy[i], input2_numpy[i])
        err_numpy += err_curr
    err_numpy /= shape4d[0]
    assert np.allclose(err, err_numpy)

    shape5d = [4, 6, 1, 32, 32]
    input, input2 = create_input_pair(shape5d, sigma=0.1)
    input_numpy = utils.torch_to_numpy(input)
    input2_numpy = utils.torch_to_numpy(input2)

    err = loss.nmse(input, input2).item()
    err_numpy = 0
    for i in range(shape5d[0]):
        err_numpy += nmse(input_numpy[i][:,0], input2_numpy[i][:,0])
    err_numpy /= shape5d[0]

    assert np.allclose(err, err_numpy)
