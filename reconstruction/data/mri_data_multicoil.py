"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import print_function

import os

import h5py
import numpy as np
import medutils
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from common import utils
import re


def _read_data(sample, keys=None, attr_keys=None, device=None):
    """ prepare the data to be in the preferred format and device """
    if keys is None:  # allow keys = {}
        keys = sample.keys()
    if attr_keys is None:  # allow attr_keys = {}
        attr_keys = sample['attrs'].keys()

    for key in keys:
        if key in [
                'target',
                'input',
                'mask',
                'kspace',
                'smaps',
                'acceleration',
                'acceleration_true',
                'input_rss_mean',
                'input_rss',
                'target_rss',
                'fg_mask',
                'fg_mask_normedsqrt'
        ]:
            sample[key] = sample[key][0]
            if device:
                sample[key] = sample[key].to(device)

        # items that cannot be moved to device
        if key in ['fname']:
            sample[key] = sample[key][0]

    for key in attr_keys:
        if key in [
                'norm',
                'mean',
                'cov',
                'ref_max',
                'rss_max',
        ]:
            sample['attrs'][key] = sample['attrs'][key].squeeze(0)
            if device:
                sample['attrs'][key] = sample['attrs'][key].to(device)

    return sample


class GeneratePatches():
    def __init__(self, patch_ny):
        self.patch_ny = patch_ny

    def __call__(self, sample):
        np_smaps = sample['smaps']
        np_kspace = sample['kspace']
        np_mask = sample['mask'][..., 0:self.patch_ny, :]

        # remove FE Oversampling
        np_smaps = medutils.mri.removeFEOversampling(np_smaps)

        # extract patch in Ny direction
        max_ny = np_smaps.shape[-2] - self.patch_ny + 1
        start_idx = np.random.randint(0, max_ny)
        start, end = start_idx, start_idx + self.patch_ny
        np_smaps = np_smaps[..., start:end, :]

        # coil-wise ifft of kspace, then patch extraction and coil-wise fft
        np_img = medutils.mri.ifft2c(np_kspace)
        np_img = medutils.mri.removeFEOversampling(np_img)
        np_kspace_new = medutils.mri.fft2c(np_img[..., start:end, :])

        # create new adjoint
        np_target_new = medutils.mri.mriAdjointOp(
            np_kspace_new,
            smaps=np_smaps,
            mask=np.ones_like(np_mask),
            coil_axis=(1),
        )

        sample['smaps'] = np_smaps
        sample['target'] = np_target_new
        sample['kspace'] = np_kspace_new
        sample['mask'] = np_mask

        if 'fg_mask' in sample.keys():
            np_fg_mask = sample['fg_mask']
            np_fg_mask = medutils.mri.removeFEOversampling(np_fg_mask, axes=(-2, -1))
            np_fg_mask = np_fg_mask[..., start:end, :]
            sample['fg_mask'] = np_fg_mask
        return sample

class LoadNorm():
    """Load Coil Sensitivities
    """
    def __init__(self, smaps_dir, num_smaps):
        assert isinstance(smaps_dir, (str))
        self.smaps_dir = smaps_dir
        self.num_smaps = num_smaps

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]

        # load smaps for current slice index
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.smaps_dir, fname),
            'r',
            libver='latest',
            swmr=True,
        )

        np_normval = []
        np_rss_max = []
        norm = sample['attrs']['norm_str']

        is_testset = 'rss_max' not in h5_data.attrs.keys()

        for i in range(sample['acceleration'].shape[0]):
            if 'acl' in sample.keys() and sample['acl'] is not None:
                acl = sample['acl']
            else:
                acl = 15 if sample['acceleration'][i] == 8 else 30

            normval = h5_data.attrs[f"{norm}_acl{acl}"]
            np_normval.append(normval)
            if not is_testset:
                np_rss_max.append(h5_data.attrs['rss_max'] / normval)

        h5_data.close()

        sample['attrs']['norm'] = np.array(np_normval)
        if not is_testset:
            sample['attrs']['rss_max'] = np.array(np_rss_max)

        return sample


class LoadCoilSensitivities():
    """Load Coil Sensitivities
    """
    def __init__(self, smaps_dir, num_smaps):
        assert isinstance(smaps_dir, str)
        self.smaps_dir = smaps_dir
        self.num_smaps = num_smaps

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]
        norm = sample['attrs']['norm_str']

        # load smaps for current slice index
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.smaps_dir, fname),
            'r',
            libver='latest',
            swmr=True,
        )

        rss_key = 'rss_max'
        is_testset = rss_key not in h5_data.attrs.keys()

        np_smaps = []
        np_target = []
        np_normval = []
        np_max = []
        np_mean = []
        np_cov = []
        np_rss_max = []

        for i in range(sample['acceleration'].shape[0]):
            # check whether sensitivity map for acl15 or acl30 is loaded
            if 'acl' in sample.keys() and sample['acl'] is not None:
                acl = sample['acl']
            else:
                acl = 15 if sample['acceleration'][i] == 8 else 30

            # training data
            # use only num_smaps set of espirit coil sensitivity maps
            smaps_sl = h5_data[f"smaps_acl{acl}"]
            np_smaps.append(utils.load_h5py_ensure_complex(
                smaps_sl[sample["slidx"][i], :, :self.num_smaps:]
            ))

            normval = h5_data.attrs[f"{norm}_acl{acl}"]
            np_normval.append(normval)
            np_mean.append(h5_data.attrs[f'lfimg_mean_acl{acl}'] / normval)
            np_cov.append(
                np.array(h5_data.attrs[f'lfimg_cov_acl{acl}']) / normval)

            if not is_testset:
                ref = h5_data[f"reference_acl{acl}"]
                np_target.append(utils.load_h5py_ensure_complex(
                    ref[sample["slidx"][i], :self.num_smaps:]
                ))
                np_max.append(
                    h5_data.attrs[f'reference_max_acl{acl}'] / normval)
                np_rss_max.append(h5_data.attrs[rss_key]/normval)

        h5_data.close()

        sample['attrs']['norm'] = np.array(np_normval)
        sample['attrs']['mean'] = np.array(np_mean)
        sample['attrs']['cov'] = np.array(np_cov)
        sample['smaps'] = np.array(np_smaps)

        if not is_testset:
            sample['target'] = np.array(np_target)
            sample['attrs']['ref_max'] = np.array(np_max)
            sample['attrs']['rss_max'] = np.array(np_rss_max)

        return sample


class GenerateCartesianMask():
    """
    Generate regular Cartesian sampling mask

    Args:
        acl (int): Number of Auto-Calibration Lines
        acc (list of ints): List of acceleration factors
    """

    def __init__(self, acl, acc, anatomy='knee'):
        assert isinstance(acl, (int))
        assert isinstance(acc, list)
        for elem in acc:
            assert isinstance(elem, (int))

        self.acl = acl
        self.acc = acc
        self.anatomy = anatomy

    def __call__(self, sample):
        np_kspace = sample["kspace"]
        nFE, nPEOS = np_kspace.shape[-2:]
        nPE = sample['attrs']['nPE']

        # parameter to mask oversampling
        padl = int(np.floor((nPEOS-nPE)/2))
        padr = int(np.ceil((nPEOS-nPE)/2))

        center = nPEOS//2
        line = np_kspace[0, 0]

        acc = np.random.choice(self.acc)
        line = np.zeros(nPEOS)

        # Match ipat pattern from scanner for knee...
        if acc == 4 and self.anatomy == 'knee':
            offset = 2
        else:
            offset = 0

        # auto-calibration lines
        line[center-self.acl//2:center+self.acl//2] = 1

        # acceleration
        if padl == 0 and padr == 0:
            line[offset::acc] = 1
            acc_true = line.size / np.sum(line)
        else:
            # mask base resolution
            line[padl+offset:-padr:acc] = 1
            line[0:padl] = 1
            line[-padr:] = 1
            acc_true = float(line[padl:-padr].size) / np.sum(line[padl:-padr])

        # expand mask to match nFE and volume dims
        np_mask = np.repeat(line[np.newaxis, ...], nFE, axis=0)
        np_mask = np.repeat(
            np_mask[np.newaxis, np.newaxis, np.newaxis, ...],
            np_kspace.shape[0],
            axis=0,
        )

        sample['mask'] = np_mask
        sample['acceleration'] = np.ones((np_kspace.shape[0])) * acc

        # compute the REAL acceleration factor
        acc_true = np.ones((np_kspace.shape[0])) * acc_true
        sample['acceleration_true'] = acc_true

        return sample


class GenerateRandomFastMRIChallengeMask:
    """
    [Code from https://github.com/facebookresearch/fastMRI]

    GenerateRandomFastMRIChallengeMask creates a sub-sampling mask of
    a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies
        2. The other columns are selected uniformly at random with a
           probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is
    equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the MaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __init__(
        self, center_fractions, accelerations,
        is_train=True, seed=None,
    ):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns
                to be retained. If multiple values are provided, then one of
                these numbers is chosen uniformly each time.

            accelerations (List[int]): Amount of under-sampling. This should
                have the same length as center_fractions. If multiple values
                are provided, then one of these is chosen uniformly each time.
                An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should'
                             'match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.is_train = is_train
        self.seed = seed

    def __call__(self, sample):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The
                shape should have at least 3 dimensions. Samples are drawn
                along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting
                the seed ensures the same mask is generated each time for the
                same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        # reset seed
        if self.seed:
            np.random.seed(self.seed)

        # extract information
        np_kspace = sample["kspace"]
        nFE, nPEOS = np_kspace.shape[-2:]
        nPE = sample['attrs']['nPE']
        np_full_mask = []
        np_acc = []
        np_acc_true = []

        data_ndim = len(np_kspace.shape[1:])
        for _ in range(np_kspace.shape[0]):
            # FAIR code starts here
            # num_cols = nPE <-- This should be the correct one...
            num_cols = nPEOS # <-- This is wrong, because lines should not be sampled in the "oversampling" regions...
            if not self.is_train:
                fname = sample['fname'].split('/file')[-1].split('.h5')[0]
                fname = fname.lower().replace('_','')
                fname = re.sub(r'[a-z]+', '', fname, re.I)
                seed = int(fname)
                np.random.seed(seed)
            choice = np.random.randint(0, len(self.accelerations))
            center_fraction = self.center_fractions[choice]
            acc = self.accelerations[choice]
            np_acc.append(acc)

            # Create the sampling line
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acc - num_low_freqs) / \
                   (num_cols - num_low_freqs)

            line = np.random.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            line[pad:pad + num_low_freqs] = True

            # FAIR code ends here

            # parameter to mask oversampling
            padl = int(np.floor((nPEOS - nPE) / 2))
            padr = int(np.ceil((nPEOS - nPE) / 2))

            if padl > 0 and padr > 0:
                # mask base resolution
                line[0:padl] = 1 # Set oversampling regions in mask to one to force data consistency to be zero there
                line[-padr:] = 1
                acc_true = line[padl:-padr].size / np.sum(line[padl:-padr]) # True acceleration factor only in *sampled* region
            else:
                acc_true = line.size / np.sum(line)
            np_acc_true.append(acc_true)

            # Reshape the mask to match the input size
            np_mask = np.repeat(line[np.newaxis, ...], nFE, axis=0)
            np_mask = np_mask.reshape((1,) * (data_ndim - 2) + np_mask.shape)

            np_full_mask.append(np_mask)

        sample['mask'] = np.array(np_full_mask)
        sample['acceleration'] = np.array(np_acc)
        sample['acceleration_true'] = np.array(np_acc_true)

        return sample


class SetupRandomFastMRIChallengeMask:
    def __call__(self, sample):
        # extract information
        np_kspace = sample["kspace"]
        nFE, nPEOS = np_kspace.shape[-2:]
        nBatch = np_kspace.shape[0]
        nPE = sample['attrs']['nPE']
        line = sample["line"]
        data_ndim = len(np_kspace.shape[1:])
        assert isinstance(line, np.ndarray)
        line = line.astype(np.float32)

        # parameter to mask oversampling
        padl = int(np.floor((nPEOS - nPE) / 2))
        padr = int(np.ceil((nPEOS - nPE) / 2))

        if padl > 0 and padr > 0:
            # mask base resolution
            line[0:padl] = 1 # Set oversampling regions in mask to one to force data consistency to be zero there
            line[-padr:] = 1
            np_acc_true = line[padl:-padr].size / np.sum(line[padl:-padr]) # True acceleration factor only in *sampled* region
        else:
            np_acc_true = line.size / np.sum(line)

        # Reshape the mask to match the input size
        np_mask = np.repeat(line[np.newaxis, ...], nFE, axis=0)
        np_mask = np_mask.reshape((1,) * (data_ndim - 2) + np_mask.shape)
        np_mask = np.repeat(np_mask[np.newaxis, ...], nBatch, axis=0)
        np_acc_true = np.repeat(np_acc_true[np.newaxis, ...], nBatch, axis=0)
        np_acc = sample['acceleration']
        np_acc = np.repeat(np_acc[np.newaxis, ...], nBatch, axis=0)
        sample['mask'] = np_mask
        sample['acceleration_true'] = np_acc_true
        sample['acceleration'] = np_acc
        return sample


class ComputeBackgroundNormalization():
    """ Estimate background mean value of RSS reconstructions. Used for challenge only."""
    def __call__(self, sample):
        np_kspace = sample["kspace_bg"]
        np_mask = sample["mask"][:, 0]  # get rid of smaps dim here

        # compute rss input for bg correction
        np_input_rss = medutils.mri.rss(
            medutils.mri.ifft2c(np_kspace * np_mask), coil_axis=1)
        # rescale wrt to *true* acceleration factor
        acc_true = sample['acceleration_true']
        np_input_rss *= np.sqrt(
            acc_true.reshape(
                len(acc_true), *[1] * len(np_input_rss.shape[1:])
            )
        )
        # extract 100x100 patch of the left upper corner of the first slice
        np_input_rss = medutils.visualization.center_crop(np_input_rss, (sample['attrs']['metadata']['rec_x'], np_input_rss.shape[-1]))[..., :100, :]
        np_input_rss_mean = np.mean(
            np_input_rss, axis=tuple(np.arange(1, len(np_input_rss.shape))))
        sample['input_rss_mean'] = np_input_rss_mean
        return sample


class LoadForegroundMask():
    def __init__(self, fg_dir):
        assert isinstance(fg_dir, (str))
        self.fg_dir = fg_dir

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.fg_dir, fname),
            'r',
            libver='latest',
            swmr=True,
        )

        sample['fg_mask'] = h5_data['foreground'][sample["slidx"]]
        h5_data.close()
        return sample

class SetupForegroundMask():
    def __call__(self, sample):
        nx, ny = sample['kspace'].shape[-2:]
        sample['fg_mask'] = np.ones((len(sample["slidx"]), nx, ny))
        return sample

class ComputeInit():
    def __init__(self, pcn=False):
        self.pcn = pcn

    def __call__(self, sample):
        """ Data should have folloing shapes:

        kspace: [nslice, nc, 1, nx, ny]
        mask: [1, 1, 1, nx, ny]
        smaps: [nslice, nc, nsmaps, nx, ny]
        target: [nslice, nsmaps, nx, ny]

        """
        np_kspace = sample["kspace"]
        np_smaps = sample["smaps"]
        np_mask = sample["mask"]

        # shift data to avoid fftshift / ifftshift in network training
        np_mask = np.fft.ifftshift(np_mask, axes=(-2, -1))
        Ny, Nx = np_kspace.shape[-2:]
        x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        adjust = (-1) ** (x + y)
        np_kspace = np.fft.ifftshift(np_kspace, axes=(-2, -1)) * adjust

        # compute rss reference for SSIM
        np_target_rss = medutils.mri.rss(
            medutils.mri.ifft2(np_kspace), coil_axis=1)

        # compute init
        np_kspace *= np_mask
        if self.pcn:
            np_input = medutils.mri.ifft2(np_kspace)[:, :, 0]
        else:
            np_input = medutils.mri.mriAdjointOpNoShift(
                np_kspace, np_smaps, np_mask, fft_axes=(-2, -1), coil_axis=(1))

        # extract norm
        norm = sample['attrs']['norm']

        def _batch_normalize(x):
            # match the shape of norm array
            return x / norm.reshape(len(norm), *[1]*len(x.shape[1:]))

        sample['mask'] = np_mask
        sample['input'] = _batch_normalize(np_input)
        sample['kspace'] = _batch_normalize(np_kspace)

        if 'input_rss_mean' in sample.keys():
            sample['input_rss_mean'] = _batch_normalize(
                sample['input_rss_mean'])

        if 'fg_mask' in sample.keys():
            sample['fg_mask_norm'] = np.sum(
                sample['fg_mask'],
                axis=tuple(range(1, len(sample['fg_mask'].shape))),
                keepdims=True,
            )
            normedsqrt = np.sqrt(
                np.maximum(1, sample['fg_mask_norm']) / (Ny * 368.))
            sample['fg_mask_normedsqrt'] = sample['fg_mask'] / normedsqrt

        if 'target' in sample.keys():
            sample['target'] = _batch_normalize(sample['target'])
            sample['target_rss'] = _batch_normalize(np_target_rss)

        return sample


class ToTensor():
    """ Convert sample ndarrays to tensors. """
    def np2th(self, arr):
        """ convert float or complex np.ndarray into torch tensor """
        if np.iscomplexobj(arr):
            arr = arr.astype(np.complex64)
        else:
            # int or float
            arr = arr.astype(np.float32)
        return utils.numpy_to_torch(arr)

    def __call__(self, sample):
        for key in ['norm', 'mean', 'cov', 'ref_max', 'rss_max']:
            if key in sample['attrs'].keys():
                sample['attrs'][key] = utils.numpy_to_torch(
                    sample['attrs'][key]).to(torch.float32)

        sample_dict = {
            "attrs": sample['attrs'],
            "fname": sample['fname'],
            "slidx": sample['slidx'],
        }

        for k in [
                'input',
                'kspace',
                'smaps',
                'target',
                'target_rss',
                'input_rss',
                'input_rss_mean',
                'acceleration',
                'acceleration_true',
        ]:
            if k in sample.keys():
                sample_dict.update({k: self.np2th(sample[k])})

        th_mask = self.np2th(sample['mask'])
        th_mask.unsqueeze_(-1)
        sample_dict.update({'mask': th_mask})

        if 'fg_mask' in sample.keys():
            sample_dict.update({"fg_mask": self.np2th(sample['fg_mask']).unsqueeze_(1),
            "fg_mask_norm": self.np2th(sample['fg_mask_norm']).unsqueeze_(1),
            "fg_mask_normedsqrt": self.np2th(sample['fg_mask_normedsqrt']).unsqueeze_(1),
            })

        return sample_dict


class MriDataset(Dataset):
    """MRI data set."""

    def __init__(
        self, csv_file, root_dir, batch_size, slices={}, data_filter={},
        transform=None, norm='max',
        full=False, acl=None, challenge='multicoil',
    ):
        """
        Args:
            csv_file (string): Path to the csv data set descirption file.
            root_dir (string): Directory with all the data.
            data_filter (dict): Dict of filter options that should be applied
                to csv_file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.full = full
        self.acl = acl
        self.challenge = challenge

        data_set = pd.read_csv(csv_file)

        # apply filter to data set
        for key in data_filter:
            if key != 'loc' and data_filter[key] is not None:
                _filter = eval(f"(data_set.{key} == data_filter['{key}'])")
                data_set = data_set[_filter]

        if 'loc' in data_filter:
            data_set = pd.DataFrame(
                data_set.loc[data_set.filename == data_filter['loc']])
        elif 'enc_y' in data_set:
            print('Discard samples with "enc_y > 380" ')
            data_set = data_set[data_set.enc_y <= 380]

        self.data_set = []
        self.full_data_set = []
        minsl = slices['min'] if 'min' in slices else 0
        for ii in range(len(data_set)):
            subj = data_set.iloc[ii]
            fname = subj.filename
            nPE = subj.nPE
            h5_data = h5py.File(os.path.join(root_dir, fname), 'r')
            kspace = h5_data['kspace']
            num_slices = kspace.shape[0]
            if 'max' in slices:
                maxsl = np.minimum(slices['max'], num_slices - 1)
            else:
                maxsl = num_slices - 1
            assert minsl <= maxsl
            assert isinstance(norm, str)
            attrs = {'nPE': nPE, 'norm_str': norm, 'metadata': subj.to_dict()}
            self.data_set += [(fname, minsl, maxsl, attrs)]
            self.full_data_set += [
                (fname, si, si, attrs)
                for si in range(minsl, maxsl)
            ]
            h5_data.close()
        if self.full:
            self.data_set = self.full_data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]

        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()

        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))

        # load the kspace data for the given slidx
        with h5py.File(
                os.path.join(self.root_dir, fname),
                'r',
                libver='latest',
                swmr=True,
        ) as data:
            np_kspace = data['kspace']
            if self.batch_size > np_kspace.shape[0]:
                np_kspace = np_kspace[:]  # faster than smart indexing
            else:
                np_kspace = np_kspace[slidx]
            np_kspace = utils.load_h5py_ensure_complex(np_kspace)
            np_kspace_bg = utils.load_h5py_ensure_complex(data['kspace'][0])

            # load extra metadata for test data
            np_line = np_acc = np_acl = None
            if 'mask' in data.keys():
                np_line = data['mask'][()]
            if 'acceleration' in data.attrs.keys():
                np_acc = data.attrs['acceleration']
            if 'num_low_frequency' in data.attrs.keys():
                np_acl = data.attrs['num_low_frequency']

        if self.challenge == 'multicoil':
            # add dimension for smaps
            np_kspace = np_kspace[:, :, np.newaxis]

        sample = {
            "kspace": np_kspace,
            "kspace_bg": np_kspace_bg,
            "line": np_line,
            "acceleration": np_acc,
            "acl": np_acl,
            "attrs": attrs,
            "slidx": slidx,
            "fname": fname,
            "rootdir": self.root_dir,
        }

        if self.acl:
            sample['acl'] = self.acl

        if self.transform:
            sample = self.transform(sample)

        return sample


class MriDatasetEval(Dataset):
    """MRI data set."""

    def __init__(
        self, csv_file, root_dir, batch_size,
        slices={}, data_filter={}, transform=None,
        norm='max', challenge='multicoil',
    ):
        """
        Args:
            csv_file (string): Path to the csv data set descirption file.
            root_dir (string): Directory with all the data.
            data_filter (dict): Dict of filter options that should be applied
              to csv_file.
            transform (callable, optional): Optional transform to be applied
              on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.challenge = challenge

        data_set = pd.read_csv(csv_file)

        # apply filter to data set
        for key in data_filter:
            if key != 'loc' and data_filter[key] not in [None]:
                _filter = eval(f"(data_set.{key} == data_filter['{key}'])")
                data_set = data_set[_filter]
        if 'loc' in data_filter:
            data_set = pd.DataFrame(
                data_set.loc[data_set.filename == data_filter['loc']])

        self.data_set = []
        minsl = slices['min'] if 'min' in slices else 0
        for ii in range(len(data_set)):
            subj = data_set.iloc[ii]
            fname = subj.filename
            nPE = subj.nPE
            h5_data = h5py.File(os.path.join(root_dir, fname), 'r')
            kspace = h5_data['kspace']
            num_slices = kspace.shape[0]
            if 'max' in slices:
                maxsl = np.minimum(slices['max'], num_slices-1)
            else:
                maxsl = num_slices - 1

            assert minsl <= maxsl
            assert isinstance(norm, str)
            attrs = {'nPE': nPE, 'norm_str': norm, 'metadata': subj.to_dict()}
            self.data_set += [
                (fname, slidx, num_slices, attrs)
                for slidx in range(minsl, num_slices, self.batch_size)
            ]
            h5_data.close()

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        fname, slidx, num_slices, attrs = self.data_set[idx]

        slidx = list(np.arange(
            slidx,
            np.minimum(slidx + self.batch_size, num_slices),
        ))
        with h5py.File(
                os.path.join(self.root_dir, fname),
                'r',
                libver='latest',
                swmr=True,
        ) as data:
            np_kspace = data['kspace']
            if self.batch_size > np_kspace.shape[0]:
                np_kspace = np_kspace[()]
            else:
                np_kspace = np_kspace[slidx]
            np_kspace_bg = data['kspace'][0]

            # float16 compatibility
            np_kspace = utils.load_h5py_ensure_complex(np_kspace)
            np_kspace_bg = utils.load_h5py_ensure_complex(np_kspace_bg)

            # load extra metadata for test data
            np_line = np_acc = np_acl = None
            if 'mask' in data.keys():
                np_line = data['mask'][()]
            if 'acceleration' in data.attrs.keys():
                np_acc = data.attrs['acceleration']
            if 'num_low_frequency' in data.attrs.keys():
                np_acl = data.attrs['num_low_frequency']

        # print("kspace loading took", time.time()-starttime)
        if self.challenge == 'multicoil':
            # add dimension for smaps
            np_kspace = np_kspace[:, :, np.newaxis]

        sample = {
            "kspace": np_kspace,
            "kspace_bg": np_kspace_bg,
            "line": np_line,
            "acceleration": np_acc,
            "acl": np_acl,
            "attrs": attrs,
            "slidx": slidx,
            "fname": fname,
            "rootdir": self.root_dir,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class LoadStatistics():
    """Load Coil Sensitivities
    """
    def __init__(self, smaps_dir):
        assert isinstance(smaps_dir, (str))
        self.smaps_dir = smaps_dir

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]

        # load smaps for current slice index
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.smaps_dir, fname),
            'r',
            libver='latest',
            swmr=True,
        )
        np_normval = []
        np_max = []
        np_mean = []
        np_cov = []
        np_rss_max = []
        norm = sample['attrs']['norm_str']

        for i in range(sample['acceleration'].shape[0]):
            acl = 15 if sample['acceleration'][i] == 8 else 30
            normval = h5_data.attrs[f"{norm}_acl{acl}"]
            np_normval.append(normval)
            np_max.append(h5_data.attrs[f'reference_max_acl{acl}'] / normval)
            np_mean.append(h5_data.attrs[f'lfimg_mean_acl{acl}'] / normval)
            np_cov.append(
                np.array(h5_data.attrs[f'lfimg_cov_acl{acl}']) / normval)
            np_rss_max.append(h5_data.attrs['rss_max'] / normval)

        np_normval = np.array(np_normval)
        np_max = np.array(np_max)
        np_mean = np.array(np_mean)
        np_cov = np.array(np_cov)
        np_rss_max = np.array(np_rss_max)
        h5_data.close()

        sample['attrs']['norm'] = np_normval
        sample['attrs']['mean'] = np_mean
        sample['attrs']['cov'] = np_cov
        sample['attrs']['ref_max'] = np_max
        sample['attrs']['rss_max'] = np_rss_max

        return sample


def create_eval_data_loaders(args, **kwargs):
    csv_val = kwargs.pop(
        'csv_eval', f'{args.csv_path}/multicoil_{args.data_split}.csv')
    data_filter = kwargs.pop('data_filter', {'type': args.acquisition})
    #print(data_filter)
    slices = kwargs.pop('slices', {})

    if args.data_split in ['train', 'val']:
        mask_func = GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=False,
        )
    else:
        mask_func = SetupRandomFastMRIChallengeMask()
    is_pcn = True if hasattr(args, 'pcn') and args.pcn else False
    data_transform = [
        mask_func,
        LoadCoilSensitivities(
            f'multicoil_{args.data_split}_espirit',
            num_smaps=args.num_smaps,
        ),
        ComputeInit(is_pcn),
        ToTensor(),
    ]

    if (hasattr(args, 'mask_bg') and args.mask_bg) or (hasattr(args, 'use_stl') and args.use_stl):
        data_transform.insert(1, ComputeBackgroundNormalization())
        data_transform.insert(2, LoadForegroundMask(f'multicoil_{args.data_split}_foreground'))

    data = MriDatasetEval(
        csv_val,
        args.data_path,
        transform=transforms.Compose(data_transform),
        batch_size=args.batch_size,  # load all slices
        slices=slices,
        data_filter=data_filter,
        norm=args.norm,
    )

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )

    if len(data_loader) == 0:
        raise ValueError('Eval dataset has length 0')

    return data_loader
