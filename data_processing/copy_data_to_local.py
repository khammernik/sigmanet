"""
Copy (reduced) data to local disk.

Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os, sys, time, h5py
import numpy as np
import pandas as pd
import medutils
from shutil import copyfile
import argparse
import pathlib

from tqdm import tqdm

def _report(operation, key, obj):
    type_str = type(obj).__name__.split(".")[-1].lower()
    print("%s %s: %s." % (operation, type_str, key))

def h5py_compatible_attributes(in_object):
    '''Are all attributes of an object readable in h5py?'''
    try:
        # Force obtaining the attributes so that error may appear
        [ 0 for at in in_object.attrs.iteritems() ]
        return True
    except:
        return False

def copy_attributes(in_object, out_object):
    '''Copy attributes between 2 HDF5 objects.'''
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value

def copy_as_float16(path1, path2, log=False):
    '''Compress a HDF5 file.
 
    :param path1: Input path
    :param path2: Output path
    :param log: Whether to print results of operations'
    :returns: A tuple(original_size, new_size)
    '''
    with h5py.File(path1, "r") as in_file, h5py.File(path2, "w") as out_file:
        for key, in_obj in in_file.items():
            print(key, in_obj)

            if in_obj.shape:
                in_obj_np = in_obj[:]
                # crop images
                if key in ['kspace']:
                    np_img = medutils.mri.ifft2c(in_obj_np, axes=(-2,-1))
                    np_img = medutils.mri.removeFEOversampling(np_img, axes=(-2, -1))
                    in_obj_np = medutils.mri.fft2c(np_img, axes=(-2,-1))
                    in_obj_np = np.ascontiguousarray(in_obj_np)
                if any([key_id for key_id in ['smaps', 'reference'] if key_id in key]):
                    in_obj_np = medutils.mri.removeFEOversampling(in_obj_np, axes=(-2, -1))
                    # # only use the 1st smap
                    # in_obj_np = np.ascontiguousarray(in_obj_np[..., 0:1, :, :])
                    in_obj_np = np.ascontiguousarray(in_obj_np)
                if np.iscomplexobj(in_obj_np):
                    if in_obj_np.dtype == np.complex128:
                        in_obj_np = in_obj_np.astype(np.complex64)
                    in_obj_np = in_obj_np.view(dtype=np.float32)
                in_obj_hp = np.float16(in_obj_np)
                out_obj = out_file.create_dataset(key, data=in_obj_hp)
            else:
                in_file.copy(key, out_file)
        copy_attributes(in_file, out_file)

        for key, out_obj in out_file.items():
            print(key, out_obj)
    return os.stat(path1).st_size, os.stat(path2).st_size

if __name__ == '__main__':
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
        '--max-files', type=int, default=None,
        help='Maximum number of files that should be copied'
    )
    parser.add_argument(
        '--csv-file', type=str, default=None,
        help='Use different csv file'
    )
    parser.add_argument(
        '--overwrite',  action='store_true',
        help='overwrite existing files',
    )
    parser.add_argument(
        '--float16',  action='store_true',
        help='copy as float16',
    )
    args = parser.parse_args()

    # create directory if for first time.
    output_dir = args.out_path / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Process {args.dataset}')
    extension = [ext for ext in ['_espirit', '_foreground', '_attrs'] if ext in args.dataset]
    extension = ' ' if len(extension) == 0  else extension[0]

    if args.csv_file:
        csv_file = args.csv_path / args.csv_file
    else:
        csv_file = f"{args.csv_path}/{args.dataset.replace(extension, '')}.csv"

    print(f'Csv file: {csv_file}')

    df = pd.read_csv(csv_file)

    for fname in tqdm(df.filename[:args.max_files]):
        file_id = fname.split('/')[-1]
        src = args.data_path / args.dataset / file_id
        dst = args.out_path / args.dataset / file_id

        if os.path.isfile(dst) and not args.overwrite:
            # print(f'{fname} exists! skip...')
            continue
        else:
            print(f'Copy {fname}...')
            print(f' {src} -> {dst}')
            t0 = time.time()
            if args.float16:
                origsize, newsize = copy_as_float16(src, dst, log=True)
            else:
                copyfile(src, dst)
            print(f'copied in {time.time() - t0:.4g} s')
