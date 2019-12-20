"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import resource
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../../')
import data.mri_data_multicoil as data_batch
from common.args import TestArgs
from common.utils import save_reconstructions
from common.mytorch.tensor_ops import center_crop
from models.sigmanet.train import build_model as build_sn_model
from common.train_template import (
    postprocess,
)


logging.basicConfig(level=logging.INFO)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_sn_model(args)
    model.load_state_dict(checkpoint['model'])
    return model


def run_sn(args, data_loader, model):
    """ Run Sigmanet """
    model.eval()
    logging.info(f'Run Sigmanet reconstruction')
    logging.info(f'Arguments: {args}')
    reconstructions = defaultdict(list)
    # keys = ['input', 'kspace', 'smaps', 'mask', 'fg_mask']
    # if args.mask_bg:
    #     keys.append('input_rss_mean')
    # attr_keys = ['mean', 'cov', 'norm']

    with torch.no_grad():
        for ii, sample in enumerate(tqdm(iter(data_loader))):
            sample = data_batch._read_data(sample, device=args.device)

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            x = model(
                sample['input'],
                sample['kspace'],
                sample['smaps'],
                sample['mask'],
                sample['attrs']
            )

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
            norm = sample['attrs']['norm'].reshape(-1, 1, 1)
            recons = recons * norm

            recons = recons.to('cpu').numpy()

            if args.debug and ii % 10 == 0:
                plt.imsave(
                    'run_sn_progress.png',
                    np.hstack(recons),
                    cmap='gray',
                )

            for bidx in range(recons.shape[0]):
                reconstructions[sample['fname']].append(
                    (sample['slidx'][bidx], recons[bidx])
                )

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    save_reconstructions(reconstructions, args.out_dir)


def main(args):
    if args.challenge == 'singlecoil':
        raise ValueError('This works only for multicoil challenge!')

    csv_eval = f'{args.csv_path}/multicoil_{args.data_split}.csv'

    data_loader = data_batch.create_eval_data_loaders(args, csv_eval=csv_eval)

    model = load_model(args.checkpoint)
    logging.info(model)

    run_sn(args, data_loader, model)


def create_arg_parser():
    parser = TestArgs()
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to the checkpoint containing trained model',
    )
    parser.add_argument(
        '--use-bg-noise-mean',
        action='store_true',
        help='If set, replace the background by the estimated bg noise mean of the RSS.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='If set, output image every 10',
    )

    parser.add_argument(
        '--pcn',
        action='store_true',
        help='If set, use Parallel Coil Network which uses 30ch input output'
        '(note: ignores data-term args and use closed form DC)',
    )
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
