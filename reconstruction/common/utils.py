"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
import pathlib
import h5py

import numpy as np
import torch


def _get_np_float_dtype(complex_dtype):
    """ Get equivalent float type given current complex dtype """
    if complex_dtype == np.complex64:
        return np.float32
    elif complex_dtype == np.complex128:
        return np.float64
    else:
        return np.float128

def _get_np_complex_dtype(float_dtype):
    """ Get equivalent complex type given current float dtype """
    if float_dtype == np.float32:
        return np.complex64
    elif float_dtype == np.float64:
        return np.complex128
    else:
        return np.complex256

def torch_to_complex_numpy(data):
    data = data.numpy()
    complex_dtype = _get_np_complex_dtype(data.dtype)
    return np.ascontiguousarray(data).view(complex_dtype).squeeze(-1)

def torch_to_complex_abs_numpy(data):
    data = data.numpy()
    return np.abs(data[..., 0] + 1j * data[..., 1])

def torch_to_numpy(data):
    return data.numpy()

def numpy_to_torch(data):
    if np.iscomplexobj(data):
        float_dtype = _get_np_float_dtype(data.dtype)
        data = np.ascontiguousarray(data[..., np.newaxis]).view(float_dtype)
    return torch.from_numpy(data)


def save_reconstructions(reconstructions, out_dir):
    """
    [Code from https://github.com/facebookresearch/fastMRI]
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def save_h5(reconstructions, out_dir, out_dict={}):
    """
    Saves the reconstructions from a model into h5 files, with additional information defined in out_dict.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        out_dict (dict of dict[str, np.array]). A dictionary containing other datasets to be stored in h5 file.
    TODO: merge with save_reconstructions.
    """
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname_orig, recons in reconstructions.items():
        fname = fname_orig.split('/')[-1]
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

            for key in out_dict.keys():
                f.create_dataset(key, data=out_dict[key][fname_orig])

def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


def load_h5py_ensure_complex(x):
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32)).view(dtype=np.complex64)
    else:
        return x

def load_h5py_ensure_float32(x):
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32))
    else:
        return x

def torch_fft2(x, normalized=True):
    """ fft on last 2 dim """
    xt = numpy_to_torch(x)
    kt = torch.fft(xt, 2, normalized)
    return torch_to_complex_numpy(kt)

def torch_ifft2(k, normalized=True):
    """ ifft on last 2 dim """
    kt = numpy_to_torch(k)
    xt = torch.ifft(kt, 2, normalized)
    return torch_to_complex_numpy(xt)

def torch_fft2c(x, normalized=True):
    """ fft2 on last 2 dim """
    x = np.fft.ifftshift(x, axes=(-2,-1))
    xt = numpy_to_torch(x)
    kt = torch.fft(xt, 2, normalized=True)
    k = torch_to_complex_numpy(kt)
    return np.fft.fftshift(k, axes=(-2,-1))

def torch_ifft2c(x, normalized=True):
    """ ifft2 on last 2 dim """
    x = np.fft.ifftshift(x, axes=(-2,-1))
    xt = numpy_to_torch(x)
    kt = torch.ifft(xt, 2, normalized=True)
    k = torch_to_complex_numpy(kt)
    return np.fft.fftshift(k, axes=(-2,-1))


class State():
    """ Class used to pass arbitrary items around """
    def __init__(self, **kwargs):
        self.iteration = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


class SimpleCacheLoop(ABC):
    def __init__(self, cache_size, cache_iter, **kwargs):
        """Initialise cache. define cache size and cache iteration. An arbitrary items
        can be passed using keyword arguments

        Issue: It seems that when data is cached, references to the file
        descriptors are not closed. For multithreading, this creates too many
        files and goes over the limit. As such, you might need to put this in
        yout main.py:

        torch.multiprocessing.set_sharing_strategy('file_system')

        Source: https://github.com/pytorch/pytorch/issues/973

        """
        self.cache = []
        self.cache_size = cache_size
        self.cache_iter = cache_iter
        for k, v in kwargs.items():
            setattr(self, k, v)

    def append(self, x):
        """ Append data to cache"""
        if len(self.cache) < self.cache_size:
            self.cache.append(x)
        else:
            self.cache[np.random.randint(self.cache_size)] = x

    @abstractmethod
    def one_loop(self):
        """ Main loop executed on cache """

    def loop(self):
        """ Execute operation n times using random data in cache """
        for _ in range(self.cache_iter):
            self.one_loop()
