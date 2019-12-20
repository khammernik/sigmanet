"""
Helper functions for complex pseudo covariance estimation

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np

def complex_pseudocovariance(data):
    """ Data variable hast to be already mean-free! 
        Operates on data x of size [nBatch, nSmaps, nFE, nPE, 2] """

    # compute number of elements
    N = data.size

    # seperate real/imaginary channel
    re = np.real(data)
    im = np.imag(data)

    # compute covariance entries. cxy = cyx        
    cxx = np.sum(re * re) / (N - 1)
    cyy = np.sum(im * im) / (N - 1)
    cxy = np.sum(re * im) / (N - 1)
    
    # Eigenvalue decomposition C = V*S*inv(V)
    # compute eigenvalues
    s1 = (cxx + cyy) / 2 - np.sqrt((cxx + cyy)**2 / 4 - cxx * cyy + cxy**2)
    s2 = (cxx + cyy) / 2 + np.sqrt((cxx + cyy)**2 / 4 - cxx * cyy + cxy**2)
    
    # compute eigenvectors
    v1x = s1 - cyy
    v1y = cxy
    v2x = s2 - cyy
    v2y = cxy
    
    # normalize eigenvectors
    norm1 = np.sqrt(np.sum(v1x * v1x + v1y * v1y))
    norm2 = np.sqrt(np.sum(v2x * v2x + v2y * v2y))

    v1x = v1x.copy() / norm1
    v1y = v1y.copy() / norm1

    v2x = v2x.copy() / norm2
    v2y = v2y.copy() / norm2
    
    # now we need the sqrt of the covariance matrix.
    # C^{-0.5} = V * sqrt(S) * inv(V)
    det = v1x * v2y - v2x * v1y
    s1 = np.sqrt(s1) / (det)
    s2 = np.sqrt(s2) / (det)
    
    cov_xx_half = v1x * v2y * s1 - v1y * v2x * s2
    cov_yy_half = v1x * v2y * s2 - v1y * v2x * s1
    cov_xy_half = v1x * v2x * (s2 - s1)
    cov_yx_half = v1y * v2y * (s1 - s2)
    return cov_xx_half, cov_xy_half, cov_yx_half, cov_yy_half

def matrix_invert(xx, xy, yx, yy): 
    """ Invert 2x2 matrix given by xx, xy, yx, yy """
    det = xx * yy - xy * yx
    return yy / det, -xy / det, -yx / det, xx / det

def normalize(x, mean, cov_xx_half, cov_xy_half, cov_yx_half, cov_yy_half):
    """ Complex normalization according to
        Trabelsi et al. Deep Complex Networks. arXiv preprint arXiv:1705.09792, 2017.
        https://arxiv.org/abs/1705.09792
    """
    x_m = x - mean
    re = np.real(x_m)
    im = np.imag(x_m)

    cov_xx_half_inv, cov_xy_half_inv, cov_yx_half_inv, cov_yy_half_inv = matrix_invert(cov_xx_half, cov_xy_half, cov_yx_half, cov_yy_half)
    x_norm_re = cov_xx_half_inv * re + cov_xy_half_inv * im
    x_norm_im = cov_yx_half_inv * re + cov_yy_half_inv * im
    img = x_norm_re + 1j * x_norm_im
    return img