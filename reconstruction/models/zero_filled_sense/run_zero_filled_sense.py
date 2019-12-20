"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import resource
import sys
sys.path.append('../../')

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import data.mri_data_multicoil as data_batch
from common.args import TestArgs
from common.mytorch import root_sum_of_squares
from common.utils import save_reconstructions
from common.mytorch.tensor_ops import center_crop
from common.train_template import (
    postprocess,
)

logging.basicConfig(level=logging.INFO)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def run_zero_filled_sense(args, data_loader):
    """ Run Adjoint (zero-filled SENSE) reconstruction """
    logging.info('Run zero-filled SENSE reconstruction')
    logging.info(f'Arguments: {args}')
    reconstructions = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(iter(data_loader)):
            sample = data_batch._read_data(sample)

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            x = sample['input']

            recons = postprocess(x, (rec_x, rec_y))

            # mask background using background mean value
            if args.mask_bg:
                fg_mask = center_crop(
                    sample['fg_mask'],
                    (rec_x, rec_y),
                ).squeeze(1)
                if args.use_bg_noise_mean:
                    bg_mean = sample['input_rss_mean'].reshape(-1, 1, 1)
                    recons = recons * fg_mask + (1 - fg_mask) * bg_mean
                else:
                    recons = recons * fg_mask

            # renormalize
            norm = sample['attrs']['norm'].numpy()[:, np.newaxis, np.newaxis]
            recons = recons.numpy() * norm

            for bidx in range(recons.shape[0]):
                reconstructions[sample['fname']].append(
                    (sample['slidx'][bidx], recons[bidx])
                )

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()}

    save_reconstructions(reconstructions, args.out_dir)


def main(args):
    if args.challenge == 'singlecoil':
        raise ValueError('This works only for multicoil challenge!')

    csv_eval = f'{args.csv_path}/multicoil_{args.data_split}.csv'

    data_loader = data_batch.create_eval_data_loaders(args, csv_eval=csv_eval)
    run_zero_filled_sense(args, data_loader)


def create_arg_parser():
    parser = TestArgs()
    parser.add_argument(
        '--use-bg-noise-mean',
        action='store_true',
        help='If set, replace the background by the estimated bg noise mean of the RSS. [Challenge only]',
    )
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
