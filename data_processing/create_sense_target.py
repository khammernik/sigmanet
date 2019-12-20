"""
Create sensitivity-weighted target

Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import h5py
import os
import medutils
import argparse
import pathlib
from tqdm import tqdm
from medutils.visualization import center_crop
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
     '--data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)
parser.add_argument(
     '--out-path', type=pathlib.Path, required=True,
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
        '--acceleration', nargs='+', default=[8, 4], type=int,
        help='Number of acceleration factors that should be considered. Same length as --acl.',
)
parser.add_argument(
     '--mask-bg', action='store_true',
     help='Mask background',
)
args = parser.parse_args()

print(f'Estimate target for {args.dataset}')

target_path = args.data_path / f'{args.dataset}_espirit'
mask_path = args.data_path / f'{args.dataset}_foreground'

# dataframe
df = pd.read_csv(args.csv_path / f'{args.dataset}.csv')

acc_pbar = tqdm(zip(args.acceleration, args.acl), leave=True)

for acc, acl in acc_pbar:
    out_path = args.out_path / f'R{acc}'
    out_path.mkdir(parents=True, exist_ok=True)
    fname_pbar = tqdm(zip(df.filename, df.rec_x, df.rec_y), leave=True)

    for fname, rec_x, rec_y in fname_pbar:
        fname_id = fname.split('/')[-1]
        fname_pbar.set_description(f'Processing {fname}')
        h5_data = h5py.File(target_path / fname_id, 'r')
        ref = h5_data[f'reference_acl{acl}'][()]
        ref = center_crop(ref, (rec_x, rec_y))
        ref = medutils.mri.rss(ref, coil_axis=1)
        if args.mask_bg:
            mask = h5py.File(mask_path / f'{fname_id}', 'r')['foreground'][()]
            mask = center_crop(mask, (rec_x, rec_y))
            ref *= mask
        
        h5_out_data = h5py.File( out_path / f'{fname_id}', 'w')
        h5_out_data.create_dataset('reconstruction_sense', data=ref)
        h5_out_data.attrs['acquisition'] = h5_data.attrs['acquisition']
        h5_out_data.close()
        h5_data.close()
