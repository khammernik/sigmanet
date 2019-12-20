"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from .complex import complex_mult, complex_mult_conj
from .fft import fft2, ifft2, fft2c, ifft2c

DICOM_OFFSET=0

def root_sum_of_squares(data, dim=0, keepdim=False, eps=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim, keepdim=keepdim) + eps)


def removeFEOversampling(data, axes=(-2,-1), dicom_offset=DICOM_OFFSET):
    """ Remove Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert data.dim() >= 2

    nFE, nPE = data.shape[axes[0]:axes[1]+1]
    if nPE != nFE:
        indices = torch.arange(int(nFE*0.25)+dicom_offset, int(nFE*0.75)+dicom_offset)
#        print(indices[0],indices[-1])
        if data.device != torch.device("cpu"):
            indices = indices.cuda()
        return data.index_select(axes[0], indices)
    else:
        return data

def addFEOversampling(data, axes=(-2,-1), dicom_offset=DICOM_OFFSET):
    """ Add Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert data.dim() >= 2

    nFE = data.shape[axes[0]]*2
    pad_u = int(nFE * 0.25 + dicom_offset)
    pad_l = int(nFE * 0.25 - dicom_offset)
    cat_shape_u = list(data.shape)
    cat_shape_l = list(data.shape)
    cat_shape_u[axes[0]] = pad_u
    cat_shape_l[axes[0]] = pad_l

    cat_u = data.new_zeros(*cat_shape_u)
    cat_l = data.new_zeros(*cat_shape_l)

    return torch.cat([cat_u, data, cat_l], dim=axes[0])

def adjointOpNoShiftOversampling(th_kspace, th_smaps, th_mask):
    th_img = torch.sum(complex_mult_conj(ifft2(th_kspace * th_mask), th_smaps), dim=-4)
    th_img = removeFEOversampling(th_img, axes=(-3,-2), dicom_offset=DICOM_OFFSET)
    return th_img

def forwardOpNoShiftOversampling(th_img, th_smaps, th_mask):
    th_img_pad = addFEOversampling(th_img, axes=(-3,-2), dicom_offset=DICOM_OFFSET).unsqueeze(-4)
    #sth_img_pad = th_img.unsqueeze(-4)
    th_kspace = fft2(complex_mult(th_img_pad.expand_as(th_smaps), th_smaps)) * th_mask
    return th_kspace

def adjointSoftSenseOpNoShift(th_kspace, th_smaps, th_mask):
    th_img = torch.sum(complex_mult_conj(ifft2(th_kspace * th_mask), th_smaps), dim=(-5))
    return th_img

def forwardSoftSenseOpNoShift(th_img, th_smaps, th_mask):
    th_img_pad = th_img.unsqueeze(-5)
    th_kspace = fft2(complex_mult(th_img_pad.expand_as(th_smaps), th_smaps)) * th_mask
    th_kspace = torch.sum(th_kspace, dim=-4, keepdim=True)
    return th_kspace

def adjointSoftSenseOp(th_kspace, th_smaps, th_mask):
    th_img = torch.sum(complex_mult_conj(ifft2c(th_kspace * th_mask), th_smaps), dim=(-5))
    return th_img

def forwardSoftSenseOp(th_img, th_smaps, th_mask):
    th_img_pad = th_img.unsqueeze(-5)
    th_kspace = fft2c(complex_mult(th_img_pad.expand_as(th_smaps), th_smaps)) * th_mask
    th_kspace = torch.sum(th_kspace, dim=-4, keepdim=True)
    return th_kspace
