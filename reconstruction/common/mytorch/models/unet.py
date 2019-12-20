"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.mytorch.tensor_ops import center_crop
from common.mytorch.init import weight_init


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each
    followed by instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, dropout=True,
                 normalize=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.normalize = normalize
        self.dropout = dropout
        self.drop_prob = drop_prob

        layers = []
        layers.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_chans))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(drop_prob))
        layers.append(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_chans))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(drop_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape
                [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape
                [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        repr_str = f'ConvBlock(in_chans={self.in_chans}'
        repr_str += f', out_chans={self.out_chans}'
        if self.dropout:
            repr_str += f', drop_prob={self.drop_prob})'
        else:
            repr_str += ')'
        return repr_str


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net:
        Convolutional networks for biomedical image segmentation.
        In International Conference on Medical image computing and
        computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(
            self,
            in_chans,
            out_chans,
            chans,
            num_pool_layers,
            drop_prob,
            pad_data=False,
            **kwargs,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
            model. chans (int): Number of output channels of the first
            convolution layer. num_pool_layers (int): Number of down-sampling
            and up-sampling layers. drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.pad_data = pad_data
        dropout = kwargs.pop('dropout', False)
        normalize = kwargs.pop('normalize', True)
        pool_mode = kwargs.pop('pool_mode', 'max')
        if pool_mode == 'mean':
            self.pooling_op = F.avg_pool2d
        else:
            self.pooling_op = F.max_pool2d

        self.down_sample_layers = nn.ModuleList([
            ConvBlock(in_chans, chans, drop_prob, dropout, normalize)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [
                ConvBlock(ch, ch * 2, drop_prob, dropout, normalize)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, dropout, normalize)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [
                ConvBlock(ch * 2, ch // 2, drop_prob, dropout, normalize)]
            ch //= 2
        self.up_sample_layers += [
            ConvBlock(ch * 2, ch, drop_prob, dropout, normalize)]
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(ch, ch // 2, kernel_size=1),
        #     nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        #     nn.Conv2d(out_chans, out_chans, kernel_size=1),
        # )
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def calculate_downsampling_padding2d(self, tensor):
        # calculate pad size
        factor = 2 ** self.num_pool_layers
        imshape = np.array(tensor.shape[-2:])
        paddings = np.ceil(imshape / factor) * factor - imshape
        paddings = paddings.astype(np.int) // 2
        p2d = (paddings[1], paddings[1], paddings[0], paddings[0])
        return p2d

    def pad2d(self, tensor, p2d):
        if np.any(p2d):
            # order of padding is reversed. that's messed up.
            tensor = F.pad(tensor, p2d)
        return tensor

    def unpad2d(self, tensor, shape):
        if tensor.shape == shape:
            return tensor
        else:
            return center_crop(tensor, shape)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape
                [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape
                [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        if self.pad_data:
            orig_shape2d = output.shape[-2:]
            p2d = self.calculate_downsampling_padding2d(output)
            output = self.pad2d(output, p2d)

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pooling_op(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(
                output,
                scale_factor=2,
                mode='bilinear',
                align_corners=False,
            )
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        output = self.conv2(output)

        if self.pad_data:
            output = self.unpad2d(output, orig_shape2d)

        return output


class InstanceNormUnetModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unet = UnetModel(*args, **kwargs)
        self.apply(weight_init)

    def forward(self, input):
        eps = 1e-12

        # output, mean, std = normalize_instance(input)
        # compute stats
        mean = input.mean(dim=(-1, -2), keepdim=True)
        std = input.std(dim=(-1, -2), keepdim=True)

        # normalise
        normalized_input = (input - mean) / (std + eps)

        # network
        output = self.unet(normalized_input)

        # unnormalise
        output = output * std + mean

        return output
