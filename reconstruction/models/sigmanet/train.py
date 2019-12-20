"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import functools
import logging
import random
import time
from collections import defaultdict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


sys.path.append('../../')  # noqa
from common.args import TrainArgs
from common.mytorch import loss as myloss
from common.mytorch.models.datalayer import (
    DataIDLayer,
    DataGDLayer,
    DataProxCGLayer,
    DataVSLayer
)
from common.mytorch.models.didn import DIDN
from common.mytorch.models.unet import UnetModel
from common.mytorch.models.sn import SensitivityNetwork
from common.mytorch.models.pcn import ParallelCoilNetwork
from common.train_template import (
    build_optim,
    define_losses,
    postprocess,
    save_image_writer,
    save_model,
)

from common.utils import State
from data import mri_data_multicoil as data_mc

# Issue: https://github.com/pytorch/pytorch/issues/973
# Increase the filesystem count for multiprocessing and h5 compatibility
torch.multiprocessing.set_sharing_strategy('file_system')

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_loss(args):
    losses = define_losses()
    l1 = losses['l1']
    ssim = losses['ssim']

    def criterion(output, target, sample, **kwargs):
        scale = kwargs.pop('scale', 1.)
        loss_l1 = l1(output, target, sample)
        loss_ssim = ssim(output, target, sample)

        loss = loss_ssim + loss_l1 * 1e-3
        loss /= scale

        return loss, loss_l1, loss_ssim

    return criterion


def create_batch_datasets(args, **kwargs):
    """ Create datasets based on kerstin's csv files """
    # make experiments (partially) random
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_train = kwargs.pop('csv_train', f'{args.csv_path}/multicoil_train.csv')
    csv_val = kwargs.pop('csv_val', f'{args.csv_path}/multicoil_val.csv')

    data_filter = kwargs.pop('data_filter', {})
    train_slices = kwargs.pop('train_slices', {'min': 5})
    test_slices = kwargs.pop('test_slices', {'min': 12, 'max': 25})

    # define transforms
    train_transforms = [
        data_mc.GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=True,
        ),
        data_mc.LoadCoilSensitivities(
            'multicoil_train_espirit',
            num_smaps=args.num_smaps,
        ),
        data_mc.ComputeBackgroundNormalization(),
        data_mc.GeneratePatches(args.fe_patch_size),
        data_mc.ComputeInit(args.pcn),
        data_mc.ToTensor(),
    ]

    val_transforms = [
        data_mc.GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=False,
        ),
        data_mc.LoadCoilSensitivities(
            'multicoil_val_espirit',
            num_smaps=args.num_smaps,
        ),
        data_mc.ComputeBackgroundNormalization(),
        data_mc.GeneratePatches(320),
        data_mc.ComputeInit(args.pcn),
        data_mc.ToTensor(),
    ]

    if args.use_fg_mask:
        train_transforms.insert(-3, data_mc.LoadForegroundMask('multicoil_train_foreground'))
        val_transforms.insert(-3, data_mc.LoadForegroundMask('multicoil_val_foreground'))
    else:
        val_transforms.insert(-3, data_mc.SetupForegroundMask()) # pseudo-fg mask with all ones
        train_transforms.insert(-3, data_mc.SetupForegroundMask())

    # create the training data set
    train_dataset = data_mc.MriDataset(
        csv_train,
        args.data_path,
        transform=transforms.Compose(train_transforms),
        batch_size=args.batch_size,  # slice numbers
        slices=train_slices,
        data_filter=data_filter,
        norm=args.norm,
        full=args.full_slices,
    )

    if len(train_dataset) == 0:
        raise ValueError('Train dataset has length 0')

    test_dataset = data_mc.MriDataset(
        csv_val,
        args.data_path,
        transform=transforms.Compose(val_transforms),
        batch_size=args.batch_size,
        slices=test_slices,
        data_filter=data_filter,
        norm=args.norm,
    )

    if len(test_dataset) == 0:
        raise ValueError('Test dataset has length 0')

    return train_dataset, test_dataset


def create_data_loaders(args, **kwargs):
    """ Create data loaders """
    train_data, dev_data = create_batch_datasets(args, **kwargs)
    display_data = [dev_data[0]]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1, # batch_size is defined in the dataset itself to overcome different nPE
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1, # batch_size is defined in the dataset itself to overcome different nPE
        num_workers=1,
        pin_memory=False,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(state, model, data_loader, optimizer, writer):
    model.train()
    args = state.args
    keys = ['input', 'target', 'kspace', 'smaps', 'mask', 'fg_mask']
    attr_keys = ['mean', 'cov', 'ref_max']
    save_image = functools.partial(save_image_writer, writer, state.epoch)
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()

    perf_avg = 0
    for iter, sample in enumerate(data_loader):
        t0 = time.perf_counter()
        sample = data_mc._read_data(sample, keys, attr_keys, args.device)
        output = model(
            sample['input'],
            sample['kspace'],
            sample['smaps'],
            sample['mask'],
            sample['attrs'],
        )

        rec_x = sample['attrs']['metadata']['rec_x']
        rec_y = sample['attrs']['metadata']['rec_y']

        output = postprocess(output, (rec_x, rec_y))

        target = postprocess(sample['target'], (rec_x, rec_y))
        sample['fg_mask'] = postprocess(sample['fg_mask'], (rec_x, rec_y))
        loss, loss_l1, loss_ssim = state.loss_fn(
            output=output,
            target=target,
            sample=sample,
            scale=1. / args.grad_acc,
        )
        t1 = time.perf_counter()
        loss.backward()
        if state.global_step % state.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
        state.global_step += 1
        t2 = time.perf_counter()
        perf = t2 - start_iter
        perf_avg += perf

        avg_loss = 0.99 * avg_loss + (0.01 if iter > 0 else 1) * loss.item()
        if iter % args.report_interval == 0:
            writer.add_scalar('TrainLoss', loss.item(), state.global_step)
            writer.add_scalar('TrainL1Loss', loss_l1.item(), state.global_step)
            writer.add_scalar(
                'TrainSSIMLoss',
                loss_ssim.item(),
                state.global_step,
            )
            logging.info(
                f'Epoch = [{state.epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f't = (tot:{perf_avg/(iter+1):.1g}s'
                f'/fwd:{t1-t0:.1g}/bwd:{t2-t1:.1g}s)'
            )

        if state.global_step % 1000 == 0:

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            input_abs = postprocess(sample['input'], (rec_x, rec_y))
            base_err = torch.abs(target - input_abs)
            pred_err = torch.abs(target - output)
            residual = torch.abs(input_abs - output)
            save_image(
                torch.cat([input_abs, output, target], -1).unsqueeze(0),
                'Train_und_pred_gt',
            )
            save_image(
                torch.cat([base_err, pred_err, residual], -1).unsqueeze(0),
                'Train_Err_base_pred',
                base_err.max(),
            )
            save_model(args, args.exp_dir, state.epoch, model, optimizer,
                       avg_loss, is_new_best=False, modelname='model_tmp.pt')

        start_iter = time.perf_counter()

#        if iter == 1000:
#            break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(state, model, data_loader, metrics, writer):
    model.eval()
    keys = ['input', 'target', 'kspace', 'smaps', 'mask', 'fg_mask']
    attr_keys = ['mean', 'cov', 'ref_max']
    losses = defaultdict(list)

    start = time.perf_counter()
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            sample = data_mc._read_data(sample, keys, attr_keys, args.device)
            output = model(
                sample['input'],
                sample['kspace'],
                sample['smaps'],
                sample['mask'],
                sample['attrs'],
            )

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            output = postprocess(output, (rec_x, rec_y))
            target = postprocess(sample['target'], (rec_x, rec_y))
            sample['fg_mask'] = postprocess(sample['fg_mask'], (rec_x, rec_y))
            loss = state.loss_fn(
                output=output,
                target=target,
                sample=sample,
                scale=1./state.grad_acc,
            )[0]
            losses['dev_loss'].append(loss.item())

            # evaluate in the foreground
            target = target.unsqueeze(1) * sample['fg_mask']
            output = output.unsqueeze(1) * sample['fg_mask']
            for k in metrics:
                losses[k].append(metrics[k](target, output, sample).item())

        for k in losses:
            writer.add_scalar(f'Dev_{k}', np.mean(losses[k]), state.epoch)

    return losses, time.perf_counter() - start


def visualize(state, model, data_loader, writer):
    save_image = functools.partial(save_image_writer, writer, state.epoch)
    keys = ['input', 'target', 'kspace', 'smaps', 'mask', 'fg_mask']
    attr_keys = ['mean', 'cov', 'ref_max']
    model.eval()
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            sample = data_mc._read_data(sample, keys, attr_keys, args.device)
            output = model(
                sample['input'],
                sample['kspace'],
                sample['smaps'],
                sample['mask'],
                sample['attrs'],
            )

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            output = postprocess(output, (rec_x, rec_y))

            input = postprocess(sample['input'], (rec_x, rec_y))
            target = postprocess(sample['target'], (rec_x, rec_y))
            fg_mask = postprocess(sample['fg_mask'], (rec_x, rec_y))

            base_err = torch.abs(target - input)
            pred_err = torch.abs(target - output)
            residual = torch.abs(input - output)
            save_image(
                torch.cat([input, output, target], -1).unsqueeze(0),
                'und_pred_gt',
            )
            save_image(
                torch.cat([base_err, pred_err, residual], -1).unsqueeze(0),
                'Err_base_pred',
                base_err.max(),
            )
            save_image(fg_mask, 'Mask', 1.)

            break


def build_model(args):
    # regularization term
    reg_config = {
        'in_chans': 2,
        'out_chans': 2,
        'pad_data': True,
    }
    if args.regularization_term == 'unet':
        reg_model = UnetModel
        reg_config.update({
            'chans': args.num_chans,
            'drop_prob': 0.,
            'normalize': False,
            'num_pool_layers': args.num_pools,
        })
    else:
        reg_model = DIDN
        reg_config.update({
            'num_chans': args.num_chans,
            'n_res_blocks': args.n_res_blocks,
            'global_residual': False,
        })

    # data term
    data_config = {
        'learnable': args.learn_data_term,
    }
    if args.data_term == 'GD':
        data_layer = DataGDLayer
        data_config.update({'lambda_init': args.lambda_init})
    elif args.data_term == 'PROX':
        data_layer = DataProxCGLayer
        data_config.update({'lambda_init': args.lambda_init})
    elif args.data_term == 'VS':
        data_layer = DataVSLayer
        data_config.update({
            'alpha_init': args.alpha_init,
            'beta_init': args.beta_init,
        })
    else:
        data_layer = DataIDLayer

    # define model
    if args.pcn:
        reg_config['in_chans'] = 30
        reg_config['out_chans'] = 30
        model = ParallelCoilNetwork(
            args.num_iter,
            reg_model,
            reg_config,
            {
                'lambda_init': args.lambda_init,
                'learnable': args.learn_data_term,
            },
            save_space=True,
            shared_params=args.shared_params,
        ).to(args.device)
    else:
        model = SensitivityNetwork(
            args.num_iter,
            reg_model,
            reg_config,
            data_layer,
            data_config,
            save_space=True,
            shared_params=args.shared_params,
        ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    #print(args.shared_params) # hack!
    #args.shared_params = False
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'], strict=False)

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def main(args):
    logging.info(args)
    # general
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    # data
    train_loader, dev_loader, display_loader = create_data_loaders(
        args,
        #csv_train=f'{args.csv_path}/multicoil_train_filtered.csv',
        #csv_val=f'{args.csv_path}/multicoil_val_two_per_scanners.csv',
        csv_train=f'{args.csv_path}/multicoil_train.csv',
        csv_val=f'{args.csv_path}/multicoil_val.csv',
        data_filter={'type': args.acquisition}
    )

    # models
    best_dev_loss = 1e9
    start_epoch = 0
    if args.checkpoint:
        logging.info('loading pretrained model...')
        checkpoint, model, optimizer = load_model(args.checkpoint)
        if args.resume:
            # args = checkpoint['args']
            best_dev_loss = checkpoint['best_dev_loss']
            start_epoch = checkpoint['epoch'] + 1
        else:
            optimizer = build_optim(args, model.parameters())
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        if args.stage_train:
            model.stage_training_init()
    optimizer.zero_grad()  # for grad accumulation

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        args.lr_step_size,
        args.lr_gamma,
    )

    # build metric functions
    _ssim = myloss.SSIM(device='cuda')
    _psnr = myloss.PSNRLoss()
    metrics = {
        'MSE': lambda x, y, kwargs: F.mse_loss(x, y),
        'NMSE': lambda x, y, kwargs: myloss.nmse(x, y),
        'PSNR': lambda x, y, s: _psnr(
            x, y, data_range=s['attrs']['ref_max']),
        'SSIM': lambda x, y, s: _ssim(
            x, y, data_range=s['attrs']['ref_max']),
    }

    # create state object
    state = State(
        epoch=0,
        global_step=0.,
        grad_acc=args.grad_acc,
        args=args,
        loss_fn=build_loss(args),
    )

    logging.info(args)
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)

    # main loop
    for epoch in range(start_epoch, args.num_epochs):
        state.epoch = epoch
        scheduler.step(state.epoch)

        train_loss, train_time = train_epoch(
            state,
            model,
            train_loader,
            optimizer,
            writer,
        )

        losses, dev_time = evaluate(
            state,
            model,
            dev_loader,
            metrics,
            writer,
        )

        visualize(state, model, display_loader, writer)
        save_key = args.save_key
        dev_loss = np.mean(losses[save_key])
        if save_key in ['SSIM', 'PSNR']:
            is_new_best = dev_loss > best_dev_loss
            best_dev_loss = max(best_dev_loss, dev_loss)
        else:
            is_new_best = dev_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model,
                   optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'TrainLoss = {train_loss:.4g} DevLoss = {dev_loss:.4g} \n  '
            f'MSE = {np.mean(losses["MSE"]):.4g}+/-'
            f'{np.std(losses["MSE"]):.4g}\n  '
            f'NMSE = {np.mean(losses["NMSE"]):.4g}+/-'
            f'{np.std(losses["NMSE"]):.4g}\n  '
            f'PSNR = {np.mean(losses["PSNR"]):.4g}+/-'
            f'{np.std(losses["PSNR"]):.4g}\n  '
            f'SSIM = {np.mean(losses["SSIM"]):.4g}+/'
            f'-{np.std(losses["SSIM"]):.4g}\n  '
            ' \n  '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s'
        )

        if args.stage_train:
            model.stage_training_transition_i(copy=False)

    writer.close()


def create_arg_parser():
    parser = TrainArgs()

    # general training settings
    parser.add_argument(
        '--resume',
        action='store_true',
        help='If set, resume the training from a previous model checkpoint. '
             '"--checkpoint" should be set with this'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to an existing checkpoint. Used along with "--resume"',
    )
    parser.add_argument(
        '--full-slices',
        action='store_true',
        help='If set, train on all slices in one epoch rather than'
             ' sampling one slice per volume',
    )
    parser.add_argument(
        '--grad-acc',
        type=int,
        default=1,
        help='Gradient accumulation',
    )
    parser.add_argument(
        '--save-key', choices=['dev_loss', 'SSIM', 'PSNR'], default='dev_loss',
        help='Save model based on best validation performance on the chosen'
        'metric',
    )

    # model hyper-parameters
    parser.add_argument(
        '--num-pools',
        type=int,
        default=3,
        help='Number of U-Net pooling layers',
    )
    parser.add_argument(
        '--n-res-blocks',
        type=int,
        default=2,
        help='Number of DownUps in DownUpBlock for DownUpNet',
    )
    parser.add_argument(
        '--num-chans',
        type=int,
        default=128,
        help='Number of initial channels',
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=1,
        help='Number of iterations for the iterative reconstruction networks',
    )
    parser.add_argument(
        '--regularization-term',
        choices=['unet', 'dunet'],
        default='dunet',
        help=(
            'Regularization term for the iterative networks. '
            'unet, dunet (down-up network)'
        )
    )
    parser.add_argument(
        '--data-term',
        choices=['GD', 'PROX', 'VS', 'NONE'],
        default='GD',
        help=(
            'Data term for the iterative networks. '
            'GD: gradient descent '
            '(Hammernik et al., Variational Network, MRM2018), '
            'PROX: conjugate gradient descent '
            '(Aggarwal et al., MoDL, TMI2018), '
            'VS: variable-splitting (Duan et al., VSNet, MICCAI2019), '
            'NONE: no data term between each iteration.')
    )

    parser.add_argument(
        '--lambda-init',
        type=float,
        default=0.1,
        help='Init of data term weight lambda.'
        'Use with "--data-term [GD|PROX]"'
    )
    parser.add_argument(
        '--alpha-init',
        type=float,
        default=0.1,
        help='Init of data term weight lambda.'
        'Use with "--data-term VS"'
    )
    parser.add_argument(
        '--beta-init',
        type=float,
        default=0.1,
        help='Init of data term weight lambda.'
        'Use with "--data-term VS"'
    )
    parser.add_argument(
        '--learn-data-term',
        action='store_true',
        help='If set, the parameters for the data term will be learnt',
    )
    parser.add_argument(
        '--pcn',
        action='store_true',
        help='If set, use Parallel Coil Network which uses 30ch input output'
        '(note: ignores data-term args and use closed form DC)',
    )
    parser.add_argument(
        '--stage-train',
        action='store_true',
        help='If set, train each cascade gradually',
    )
    parser.add_argument(
        '--shared-params',
        action='store_true',
        help='If set, share the params along iterations',
    )
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
