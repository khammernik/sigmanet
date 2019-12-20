"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim

import matplotlib.pyplot as plt
import pandas as pd
import os


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    args = parser.parse_args()
    tgt_files = sorted(args.predictions_path.iterdir())
    # filter directory to only get .h5py
    tgt_files = [ f for f in tgt_files if '.h5' in f.name ]
    for tgt_file in tgt_files:
        with h5py.File(args.predictions_path / tgt_file.name, 'r') as recons:
            # solve jinming stuff
            imkey = 'reconstruction' if 'reconstruction' in recons.keys() else 'finetun_rss'
            recons = recons[imkey][:]

        imname = args.predictions_path / f'{str(tgt_file).split("/")[-1].split(".")[0]}.png'
        plt.imsave(imname, np.hstack(recons[::5]), cmap='gray', vmax=recons.max())
