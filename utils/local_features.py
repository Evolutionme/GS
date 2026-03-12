#
# Local distribution features for 3DGS plane regularization and detail preservation.
# Provides KNN queries, feature extraction (Si, Ui, Pi), and adaptive MLP.
#

import torch
import torch.nn as nn
import math

# Prefer pytorch3d CUDA KNN; fall back to brute-force torch.cdist
try:
    from pytorch3d.ops import knn_points
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


def compute_knn(xyz, k=8, batch_size=4096):
    """Compute K-nearest neighbor indices for point cloud.

    Args:
        xyz: (N, 3) tensor of 3D positions
        k: number of neighbors
        batch_size: batch size for chunked computation to avoid OOM (fallback only)

    Returns:
        (N, k) tensor of neighbor indices
    """
    if PYTORCH3D_AVAILABLE:
        # CUDA-accelerated KNN — millisecond-level, no N×N matrix
        _, idx, _ = knn_points(
            xyz.unsqueeze(0),  # (1, N, 3)
            xyz.unsqueeze(0),  # (1, N, 3)
            K=k + 1            # +1 because result includes self
        )
        return idx[0, :, 1:]  # (N, k), drop self

    # Fallback: batched brute-force
    N = xyz.shape[0]
    if N <= batch_size:
        dists = torch.cdist(xyz.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
        dists.fill_diagonal_(float('inf'))
        _, indices = dists.topk(k, dim=1, largest=False)
        return indices
    else:
        all_indices = []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_xyz = xyz[i:end]
            dists = torch.cdist(batch_xyz.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
            for j in range(end - i):
                dists[j, i + j] = float('inf')
            _, indices = dists.topk(k, dim=1, largest=False)
            all_indices.append(indices)
        return torch.cat(all_indices, dim=0)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to rotation matrix.

    Args:
        q: (N, 4) quaternion tensor

    Returns:
        (N, 3, 3) rotation matrix tensor
    """
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class LocalFeatureExtractor:
    """Extract 3D local distribution features: Si, Ui, Pi."""

    @staticmethod
    def compute_Si(xyz, rotations, scaling, knn_idx):
        """Compute plane similarity feature Si.

        Normal direction = rotation matrix column corresponding to the smallest scaling axis.
        Si = 1 - Var(neighbor_angles) / (pi/2)

        Args:
            xyz: (N, 3) positions
            rotations: (N, 4) quaternions
            scaling: (N, 3) activated scaling values
            knn_idx: (N, k) neighbor indices

        Returns:
            (N, 1) Si values in [0, 1]
        """
        N = xyz.shape[0]
        # Build rotation matrices
        R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

        # Extract normal as the column corresponding to the smallest scaling axis
        min_scale_idx = scaling.argmin(dim=1)  # (N,)
        normals = R[torch.arange(N, device=R.device), :, min_scale_idx]  # (N, 3)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        # Get neighbor normals
        neighbor_normals = normals[knn_idx]  # (N, k, 3)

        # Compute angles between each point's normal and its neighbors' normals
        # cos_angle = |dot(n_i, n_j)| (absolute value because normals can be flipped)
        cos_angles = torch.abs(
            (normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)
        ).clamp(0.0, 1.0)  # (N, k)
        angles = torch.acos(cos_angles.clamp(max=1.0 - 1e-6))  # (N, k) in [0, pi/2]

        # Variance of angles
        angle_var = angles.var(dim=1, keepdim=True)  # (N, 1)

        # Si = 1 - Var(angle) / (pi/2)
        Si = (1.0 - angle_var / (math.pi / 2.0)).clamp(0.0, 1.0)
        return Si

    @staticmethod
    def compute_Ui(xyz, knn_idx):
        """Compute neighborhood uniformity feature Ui.

        Ui = 1 - Std(neighbor_distances) / R_box

        Args:
            xyz: (N, 3) positions
            knn_idx: (N, k) neighbor indices

        Returns:
            (N, 1) Ui values in [0, 1]
        """
        # Get neighbor positions
        neighbor_xyz = xyz[knn_idx]  # (N, k, 3)

        # Compute distances to neighbors
        dists = torch.norm(neighbor_xyz - xyz.unsqueeze(1), dim=-1)  # (N, k)

        # Standard deviation of distances
        dist_std = dists.std(dim=1, keepdim=True)  # (N, 1)

        # R_box: smallest dimension of the scene bounding box
        bbox_min = xyz.min(dim=0).values
        bbox_max = xyz.max(dim=0).values
        bbox_size = bbox_max - bbox_min
        R_box = bbox_size.min().clamp(min=1e-6)

        # Ui = 1 - Std(dist) / R_box
        Ui = (1.0 - dist_std / R_box).clamp(0.0, 1.0)
        return Ui

    @staticmethod
    def compute_Pi(scaling, rotations, viewpoint_camera):
        """Compute projection pixel ratio feature Pi.

        Estimates the 2D projected area of each Gaussian and normalizes by the average.

        Args:
            scaling: (N, 3) activated scaling values
            rotations: (N, 4) quaternions
            viewpoint_camera: camera object with FoVx, FoVy, image_width, image_height, etc.

        Returns:
            (N, 1) Pi values in [0, 1]
        """
        # Approximate 2D projected area:
        # Use the two largest scaling axes (the "disk" area of the ellipsoid)
        # Sort scaling to get the two largest
        sorted_scales, _ = scaling.sort(dim=-1, descending=True)
        # Approximate ellipse area = pi * s1 * s2
        area_3d = math.pi * sorted_scales[:, 0] * sorted_scales[:, 1]  # (N,)

        # Rough projection scaling factor based on focal length
        fx = viewpoint_camera.image_width / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
        fy = viewpoint_camera.image_height / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
        focal_scale = (fx * fy)  # scalar

        # Projected pixel area (rough estimate, ignoring depth for simplicity)
        pixel_area = area_3d * focal_scale  # (N,)

        # Normalize by mean
        mean_area = pixel_area.mean().clamp(min=1e-6)
        Pi = (pixel_area / mean_area).unsqueeze(-1).clamp(0.0, 1.0)  # (N, 1)
        return Pi


class AdaptiveMLP(nn.Module):
    """Lightweight MLP: 3 -> 16 -> 8 -> 1 with ReLU + Sigmoid.

    Total parameters: (3*16+16) + (16*8+8) + (8*1+1) = 64 + 136 + 9 = 209
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training start."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Args:
            features: (N, 3) tensor of [Si, Ui, Pi]

        Returns:
            (N, 1) alpha_i values in [0, 1]
        """
        return self.net(features)
