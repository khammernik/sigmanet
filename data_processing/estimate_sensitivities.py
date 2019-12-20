"""
Coil sensitivity estimation using ESPIRiT from the bart toolbox [1]

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

[1] Uecker, M. , Lai, P. , Murphy, M. J., Virtue, P. , Elad, M. , 
    Pauly, J. M., Vasanawala, S. S. and Lustig, M. (2014), 
    ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI:
    Where SENSE meets GRAPPA.
    Magn. Reson. Med., 71: 990-1001. doi:10.1002/mrm.24751
"""

import pandas as pd
import h5py
import os
import medutils
import numpy as np
from tqdm import tqdm
import argparse
import pathlib

import helper

parser = argparse.ArgumentParser()
parser.add_argument(
     '--data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)
parser.add_argument(
     '--csv-path', type=pathlib.Path, required=True,
     help='Path to the csv files',
)
parser.add_argument(
     '--dataset', type=str, required=True,
     help='Dataset for which the csv file should be generated.'
)
parser.add_argument(
        '--acl', nargs='+', default=[15, 30], type=int,
        help='Number of low frequencies (auto-calibration lines) that should be considered \
              for coil sensitivity map estimation. Only used for train and validation datasets.',
)
parser.add_argument(
     '--num-smaps', type=int, default=2,
     help='Number of sensitivity maps for softSENSE [1]'
)
args = parser.parse_args()

print(f'Estimate sensitivities for {args.dataset}')

sensitivities_dir = args.data_path / f'{args.dataset}_espirit'
sensitivities_dir.mkdir(parents=True, exist_ok=True)

print(f'Sensitivities dir: {sensitivities_dir}')

# attributes to be copied from the original raw data files to the sensitivity map files
attrs = ['patient_id', 'acquisition']

# dataframe
df = pd.read_csv(args.csv_path / f'{args.dataset}.csv')

is_testset = any(sstr in args.dataset for sstr in ['test', 'challenge'])
print('Test/Challenge dataset: ', is_testset)

fname_pbar = tqdm(zip(df.filename, df.rec_x, df.rec_y), leave=True)
for fname, rec_x, rec_y in fname_pbar:
    fname_pbar.set_description(f'Processing {fname}')
    h5_data = h5py.File(args.data_path / fname, 'r')
    acl_list = [h5_data.attrs['num_low_frequency']] if 'num_low_frequency' in h5_data.attrs.keys() else args.acl

    file_id = fname.split('/')[-1]
    
    h5_smaps = h5py.File(sensitivities_dir / file_id, 'a', swmr=True)

    # get the kspace
    kspace = h5_data['kspace'].value
    nSl, nCh, nFE, nPEOS = kspace.shape

    acl_pbar = tqdm(acl_list, leave=True)
    for acl in acl_pbar:
        acl_pbar.set_description(f'  acl={acl}')

        # copy ISMRMRD header
        if 'ismrmrd_header' not in h5_smaps.keys():
            h5_smaps.create_dataset('ismrmrd_header', data=h5_data['ismrmrd_header'])
            # copy attributes
            for key in attrs:
                h5_smaps.attrs[key] = h5_data.attrs[key]

        # Estimate sensitivity maps using ESPIRiT        
        if not f'smaps_acl{acl}' in h5_smaps.keys():
            smaps = np.zeros(kspace.shape + (args.num_smaps,), dtype=kspace.dtype)
            slice_pbar = tqdm(range(nSl))
            for i in slice_pbar:
                slice_pbar.set_description(f'    compute smaps for slice={i}')
                kspace_sl = np.transpose(kspace[i], (1,2,0))[np.newaxis,...]
                smaps_sl = medutils.bart(1, f'ecalib -d0 -m{args.num_smaps} -r{acl}', kspace_sl)
                if args.num_smaps == 1:
                    smaps_sl = smaps_sl[..., np.newaxis]
                smaps[i] = np.transpose(smaps_sl[0], (2, 0, 1, 3))

            smaps = np.transpose(smaps, (0, 1, 4, 2, 3))
            h5_smaps.create_dataset(f'smaps_acl{acl}', data=smaps)
        else:
            smaps = h5_smaps[f'smaps_acl{acl}'].value

        # compute mask with auto-calibration lines (ACLs)
        center = nPEOS//2
        line = np.zeros(nPEOS)
        line[center-acl//2:center+acl//2] = 1
        mask = np.repeat(line[np.newaxis,...], nFE, axis=0)

        # compute normalizations on low frequency image (ACL only)
        # this is only computed on the first set of smaps because with RSS combination, we loose the complex-valued information!
        lf_img = medutils.mri.mriAdjointOp(kspace[:,:,np.newaxis,...], smaps, mask, coil_axis=1)
        lf_img = medutils.visualization.center_crop(lf_img[:,0], (rec_x, rec_y))

        # intensity normalizations
        lf_max = np.max(np.abs(lf_img))
        lf_med = medutils.mri.estimateIntensityNormalization(np.abs(lf_img))
        
        # stats for complex instance norm
        mean = np.mean(lf_img)
        cov_xx, cov_xy, cov_yx, cov_yy = helper.complex_pseudocovariance(lf_img - mean)
        
        # create attributes
        h5_smaps.attrs[f'lfimg_max_acl{acl}'] = lf_max
        h5_smaps.attrs[f'lfimg_med_acl{acl}'] = lf_med
        h5_smaps.attrs[f'lfimg_mean_acl{acl}'] = mean
        h5_smaps.attrs[f'lfimg_cov_acl{acl}'] = [cov_xx, cov_xy, cov_yx, cov_yy]

        if not is_testset:
            if not f'reference_acl{acl}' in h5_smaps.keys():
                reference = medutils.mri.mriAdjointOp(kspace[:,:,np.newaxis,...], smaps, np.ones_like(smaps[0,0,0]), coil_axis=-4)
                h5_smaps.create_dataset(f'reference_acl{acl}', data=reference)
            else:
                reference = h5_smaps[f'reference_acl{acl}'].value

            # compute max value in sensitivity-combined reference
            reference_proc = medutils.mri.rss(medutils.visualization.center_crop(reference, (rec_x, rec_y)), coil_axis=1)
            h5_smaps.attrs[f'reference_max_acl{acl}'] = np.max(reference_proc)

    if not is_testset:
        # rss_max in images
        coil_imgs = medutils.mri.ifft2c(kspace)
        rss = medutils.mri.rss(coil_imgs, coil_axis=1)
        rss = medutils.visualization.center_crop(rss, (rec_x, rec_y))
        rss_max = np.max(rss)
        h5_smaps.attrs['rss_max'] = rss_max

    h5_data.close()
    h5_smaps.close()
