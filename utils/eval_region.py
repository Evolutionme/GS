#
# Region-specific evaluation metrics for 3DGS plane regularization.
# Uses Sobel edge detection to auto-segment plane/detail regions.
#

import torch
import torch.nn.functional as F


def compute_region_mask(gt_image, grad_threshold=0.1):
    """Auto-segment plane/detail regions using image gradient.

    Args:
        gt_image: (3, H, W) tensor, values in [0, 1]
        grad_threshold: gradient threshold for segmentation

    Returns:
        plane_mask: (1, H, W) boolean tensor (True = plane region)
        detail_mask: (1, H, W) boolean tensor (True = detail region)
    """
    # Convert to grayscale
    gray = gt_image.mean(dim=0, keepdim=True)  # (1, H, W)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=gt_image.dtype, device=gt_image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=gt_image.dtype, device=gt_image.device).unsqueeze(0).unsqueeze(0)

    # Compute gradients
    grad_x = F.conv2d(gray.unsqueeze(0), sobel_x, padding=1).squeeze(0)
    grad_y = F.conv2d(gray.unsqueeze(0), sobel_y, padding=1).squeeze(0)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (1, H, W)

    # Segment
    plane_mask = gradient_magnitude < grad_threshold
    detail_mask = ~plane_mask

    return plane_mask, detail_mask


def plane_mse(rendered, gt, plane_mask):
    """Plane region MSE: measures noise suppression in flat areas.

    Args:
        rendered: (3, H, W) rendered image
        gt: (3, H, W) ground truth image
        plane_mask: (1, H, W) boolean mask

    Returns:
        scalar MSE value (lower is better)
    """
    if plane_mask.sum() == 0:
        return torch.tensor(0.0, device=rendered.device)
    diff = (rendered - gt) ** 2
    masked_diff = diff * plane_mask.float()
    return masked_diff.sum() / (plane_mask.sum() * 3)


def detail_ssim(rendered, gt, detail_mask, window_size=11):
    """Detail region SSIM: measures detail preservation in textured areas.

    Args:
        rendered: (3, H, W) rendered image
        gt: (3, H, W) ground truth image
        detail_mask: (1, H, W) boolean mask
        window_size: SSIM window size

    Returns:
        scalar SSIM value (higher is better)
    """
    from utils.loss_utils import ssim
    if detail_mask.sum() == 0:
        return torch.tensor(1.0, device=rendered.device)
    # Apply mask
    rendered_masked = rendered * detail_mask.float()
    gt_masked = gt * detail_mask.float()
    return ssim(rendered_masked, gt_masked)
