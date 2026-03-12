#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def regularization_loss(xyz, knn_idx, alpha_i, sigmoid_sharpness=10.0):
    """L_Reg: Encourage uniform spacing in plane regions (high alpha).

    For Gaussians with high alpha_i (plane-like), penalize non-uniform 
    neighbor distances to promote smooth surfaces.

    Args:
        xyz: (N, 3) Gaussian positions
        knn_idx: (N, k) neighbor indices
        alpha_i: (N, 1) adaptive coefficients
        sigmoid_sharpness: sharpness of the soft weight

    Returns:
        scalar loss value
    """
    # Soft weight for plane regions
    w_plane = torch.sigmoid(sigmoid_sharpness * (alpha_i - 0.5))  # (N, 1)

    # Compute neighbor distances
    neighbor_xyz = xyz[knn_idx]  # (N, k, 3)
    dists = torch.norm(neighbor_xyz - xyz.unsqueeze(1), dim=-1)  # (N, k)

    # Standard deviation of distances (want this to be small for planes)
    dist_std = dists.std(dim=1, keepdim=True)  # (N, 1)

    # Weighted loss: only penalize plane regions
    loss = (w_plane * dist_std).mean()
    return loss


def detail_loss(xyz, knn_idx, alpha_i, sigmoid_sharpness=10.0):
    """L_Detail: Encourage tight clustering in detail regions (low alpha).

    For Gaussians with low alpha_i (detail-like), penalize distance
    to neighborhood center to keep detail Gaussians clustered.

    Args:
        xyz: (N, 3) Gaussian positions
        knn_idx: (N, k) neighbor indices
        alpha_i: (N, 1) adaptive coefficients
        sigmoid_sharpness: sharpness of the soft weight

    Returns:
        scalar loss value
    """
    # Soft weight for detail regions
    w_detail = 1.0 - torch.sigmoid(sigmoid_sharpness * (alpha_i - 0.5))  # (N, 1)

    # Compute neighborhood center
    neighbor_xyz = xyz[knn_idx]  # (N, k, 3)
    center = neighbor_xyz.mean(dim=1)  # (N, 3)

    # Distance from each Gaussian to its neighborhood center
    dist_to_center = torch.norm(xyz - center, dim=-1, keepdim=True)  # (N, 1)

    # Weighted loss: only penalize detail regions
    loss = (w_detail * dist_to_center).mean()
    return loss
