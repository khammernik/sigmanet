"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import shutil

import torch
import torchvision

from common import mytorch
from common.mytorch.tensor_ops import center_crop


def postprocess(tensor, shape):
    """Postprocess the tensor to be magnitude image and crop to the ROI,
    which is (min(nFE, shape[0]), min(nPE, shape[1]). The method expects either a
    tensor representing complex values
    (with shape [bs, nsmaps, nx, ny, 2])
    or a real-valued tensor
    (with shape [bs, nsmaps, nx, ny])

    """
    if tensor.shape[-1] == 2:
        tensor = mytorch.mri.root_sum_of_squares(tensor, dim=(1, -1), eps=1e-9)
    cropsize = (min(tensor.shape[-2], shape[0]), min(tensor.shape[-1], shape[1]))
    return center_crop(tensor, cropsize)


def define_losses():
    _ssim = mytorch.loss.SSIM(device='cuda')

    def mse(x, xgt, sample):
        tmp = (x - xgt) * sample['fg_mask_normedsqrt'].cuda()
        loss = (tmp ** 2).sum()
        return loss / xgt.size()[0]

    def l1(x, xgt, sample):
        tmp = abs(x - xgt) * sample['fg_mask'].cuda()
        loss = tmp.sum()
        return loss / xgt.size()[0]

    def ssim(x, xgt, sample):
        SSIM_SCALE = 100
        batchsize, nFE, nPE = xgt.size()
        mask = sample['fg_mask'].cuda()
        dynamic_range = sample['attrs']['ref_max'].cuda()
        _, ssimmap = _ssim(
            xgt.view(batchsize, 1, nFE, nPE),
            x.view(batchsize, 1, nFE, nPE),
            data_range=dynamic_range, full=True,
        )

        # only take the mean over the foreground
        ssimmap = ssimmap.view(batchsize, -1)
        mask = mask.contiguous().view(batchsize, -1)
        mask_norm = mask.sum(-1, keepdim=True)
        mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))
        ssim_val = (ssimmap * mask).sum(-1) / mask_norm
        return (1 - ssim_val.mean()) * SSIM_SCALE

    def gradient_loss(x, xgt, sample):
        tmp = (x - xgt) * sample['fg_mask_normedsqrt'].cuda()
        grad_x = tmp[..., 1:] - tmp[..., :-1]
        grad_y = tmp[..., 1:, :] - tmp[..., :-1, :]
        loss = ((grad_x ** 2).sum() + (grad_y ** 2).sum())
        return loss / xgt.size()[0]

    return {
        'mse': mse,
        'l1': l1,
        'ssim': ssim,
        'gradient': gradient_loss,
    }


def build_optim(args, params):
    """ build optimizer """
    if hasattr(args, 'optimizer'):
        if args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                params,
                args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params,
                args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == 'radam':
            optimizer = mytorch.optim.RAdam(
                params,
                args.lr,
                weight_decay=args.weight_decay,
            )
    else:
        # default optimizer for now
        optimizer = torch.optim.RMSprop(
            params,
            args.lr,
            weight_decay=args.weight_decay,
        )

    return optimizer


def save_model(
    args, exp_dir, epoch, model, optimizer,
    best_dev_loss, is_new_best, modelname='model.pt',
):
    """ save model & optimizer state """
    save_dict = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_dev_loss': best_dev_loss,
        'exp_dir': exp_dir
    }
    torch.save(save_dict, f=exp_dir / modelname)
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def save_image_writer(writer, epoch, image, tag, max_val=None):
    image -= image.min()
    if max_val is not None:
        image /= max_val
    else:
        image /= image.max()
    grid = torchvision.utils.make_grid(image, nrow=2, pad_value=1)
    writer.add_image(tag, grid, epoch)
