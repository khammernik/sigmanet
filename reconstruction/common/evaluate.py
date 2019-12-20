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

from tqdm import tqdm

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)

MAP_METRICS = {'all' : 'ALL'.ljust(5), 
                'CORPD_FBK' : 'PD'.ljust(5),
                'CORPDFS_FBK' : 'PDFS'.ljust(5),
                'AXFLAIR' : 'AXFLAIR',
                'AXT1POST' : 'AXT1POST',
                'AXT1PRE' : 'AXT1PRE',
                'AXT1' : 'AXT1',
                'AXT2' : 'AXT2'}

KNEE_ACQ = ['CORPD_FBK', 'CORPDFS_FBK']
BRAIN_ACQ = ['AXFLAIR', 'AXT1POST', 'AXT1PRE', 'AXT1', 'AXT2']


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }
        self.metrics_numpy = {metric: [] for metric in metric_funcs}
        self.metrics_info = {'filename' : [], 'acquisition' : []}

    def push(self, target, recons, filename, acquisition):
        self.metrics_info['filename'].append(filename)
        self.metrics_info['acquisition'].append(acquisition)
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))
            self.metrics_numpy[metric].append(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def individual_eval(self):
        return {**self.metrics_info, **self.metrics_numpy}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )

    def latex(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        metric_names = [name for name in metric_names if name != 'MSE']
        print('Metrics: ', metric_names)
        outstr = ''
        for name in metric_names:
            if name == 'PSNR':
                outstr += f'{means[name]:.2f} $\\pm$ {2 * stddevs[name]:.2f}'
            elif name != 'MSE':
                outstr += f'{means[name]:.4f} $\\pm$ {2 * stddevs[name]:.4f}'

            if name != metric_names[-1]:
                outstr += ' & '
            else:
                outstr += ' \\\\'
        print(outstr)


class VerboseMetrics(Metrics):
    def push(self, target, recons, filename, acquisition):
        self.metrics_info['filename'].append(filename)
        self.metrics_info['acquisition'].append(acquisition)
        metric_str = ""
        for metric, func in METRIC_FUNCS.items():
            res = func(target, recons)
            metric_str += f'{metric}: {res:.4g} '
            self.metrics[metric].push(res)
            self.metrics_numpy[metric].append(res)
        print(metric_str)

def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    # If partial, filter data
    tgt_files = sorted(args.target_path.iterdir())
    if args.partial:
        prediction_files = [ x.name.split('/')[-1] for x in list(args.predictions_path.iterdir())]
        tgt_files = [ x for x in tgt_files if x.name.split('/')[-1] in prediction_files ]

    for tgt_file in tqdm(tgt_files):
        with h5py.File(tgt_file, 'r') as target, h5py.File(
                args.predictions_path / tgt_file.name, 'r') as recons:
            target_acquisition = target.attrs['acquisition']
            if args.acquisition and args.acquisition != target_acquisition:
                continue
            #print(tgt_file)
            target = target[recons_key].value
            recons = recons['reconstruction'].value
            metrics.push(target, recons, tgt_file.name.split('/')[-1], target_acquisition)
            imname = args.predictions_path / f'{str(tgt_file).split("/")[-1].split(".")[0]}.png'
            diff = abs(np.hstack(target[::5] - recons[::5]))
            im = np.vstack([
                np.hstack(target[::5]),
                np.hstack(recons[::5]),
                diff / abs(diff).max() * target.max(),
            ])
            plt.imsave(imname, im, cmap='gray')

    return metrics

def latex_individual(df, fname, dataset):

    f = open(fname, 'w')
    if dataset == 'knee':
        acq_list = ['all', ] + KNEE_ACQ
    elif dataset == 'brain':
        acq_list = ['all',] + BRAIN_ACQ
    else:
        raise ValueError(f'Dataset {dataset} is not defined!')
    for acq in acq_list:
        if args.acquisition and args.acquisition != acq:
            continue
        df_eval = df[df.acquisition == acq].reset_index(drop=True) if acq != 'all' else df
        outstr = f'{MAP_METRICS[acq]} & '
        for name in METRIC_FUNCS.keys():
            if name == 'MSE': continue
            current_mean = df_eval[name].mean()
            current_std  = df_eval[name].std()
            
            if name == 'PSNR':
                outstr += f'{current_mean:.2f} $\\pm$ {current_std:.2f}'
            elif name != 'MSE':
                outstr += f'{current_mean:.4f} $\\pm$ {current_std:.4f}'

            if name != list(METRIC_FUNCS.keys())[-1]:
                outstr += ' & '
            else:
                outstr += ' \\\\'
        print(outstr)
        f.write(outstr)
    f.close()
    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                        help='Which challenge')
    parser.add_argument('--sense-target', action='store_true',
                        help='If set, only evaluate for the existing reconstructions')
    parser.add_argument('--acquisition', choices=KNEE_ACQ + BRAIN_ACQ, default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')
    parser.add_argument('--dataset', choices=['knee', 'brain'], required=True,
                        help='Which challenge')
    parser.add_argument('--partial', action='store_true',
                        help='If set, only evaluate for the existing reconstructions')
    parser.add_argument('--save-im', action='store_true',
                        help='If set, save image at the recon directory')
    args = parser.parse_args()

    if args.challenge == 'singlecoil':
        recons_key = 'reconstruction_esc'
    elif args.challenge == 'multicoil' and args.sense_target:
        recons_key = 'reconstruction_sense'
    else:
        recons_key = 'reconstruction_rss'

    print(args.predictions_path)
    metrics = evaluate(args, recons_key)
    print(metrics)
    print('')
    print('Latex:')
    metrics.latex()

    df = pd.DataFrame(data=metrics.individual_eval())

    if args.dataset == 'knee':
        acq_list = ['all', ] + KNEE_ACQ
    elif args.dataset == 'brain':
        acq_list = ['all',] + BRAIN_ACQ
    else:
        raise ValueError(f'Dataset {args.dataset} is not defined!')
    
    for acq in acq_list:
        if args.acquisition and args.acquisition != acq:
            continue
        df_acq = df[df.acquisition == acq].reset_index(drop=True)
        df_acq.to_csv(os.path.join(args.predictions_path, f'eval_{acq}.csv'), index=False)

    latex_individual(df, os.path.join(args.predictions_path, 'eval_latex.txt'), args.dataset)

    if not args.acquisition:
        df.to_csv(os.path.join(args.predictions_path, 'eval_all.csv'), index=False)
