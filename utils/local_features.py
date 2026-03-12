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
        """各向异性平面度特征 S_hat_i。
        
        计算邻域高斯中心的 3×3 协方差矩阵，做特征分解，
        用 planarity = (λ2 - λ3) / λ1 衡量局部几何的平面程度。
        
        - S_hat_i → 1：邻域点分布在一个平面上（λ3 ≈ 0）
        - S_hat_i → 0：邻域点各向同性分布或线性分布
        """
        N = xyz.shape[0]
        
        # 获取邻域坐标
        neighbor_xyz = xyz[knn_idx]                          # (N, k, 3)
        
        # 中心化：减去邻域中心
        center = neighbor_xyz.mean(dim=1, keepdim=True)      # (N, 1, 3)
        centered = neighbor_xyz - center                     # (N, k, 3)
        
        # 计算 3×3 协方差矩阵: C = (1/k) * X^T X
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (N, 3, 3)
        cov = cov / knn_idx.shape[1]
        
        # 特征分解（对称矩阵，用 symeig/linalg.eigh）
        eigenvalues, _ = torch.linalg.eigh(cov)              # (N, 3), 升序
        # eigh 返回升序：λ_small, λ_mid, λ_large
        lam3 = eigenvalues[:, 0].clamp(min=1e-8)   # λ3 (最小)
        lam2 = eigenvalues[:, 1].clamp(min=1e-8)   # λ2 (中间)
        lam1 = eigenvalues[:, 2].clamp(min=1e-8)   # λ1 (最大)
        
        # 平面度: planarity = (λ2 - λ3) / λ1
        Si = ((lam2 - lam3) / lam1).clamp(0.0, 1.0).unsqueeze(-1)  # (N, 1)
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
    def compute_Vi(visibility_history):
        """多视角可见性方差 V_i。
        
        衡量高斯在最近 K 个视角下可见性的一致程度。
        
        - V_i → 0：可见性稳定（全可见或全不可见），通常是平面内部
        - V_i → 1：可见性不稳定（某些视角可见某些不可见），通常是边缘/遮挡处
        
        Args:
            visibility_history: (N, K) tensor, 0/1 值，最近 K 个视角的可见性
    
        Returns:
            (N, 1) Vi values in [0, 1]
        """
        # Bernoulli 方差 = p * (1 - p)，最大值 0.25
        mean_vis = visibility_history.float().mean(dim=1, keepdim=True)   # (N, 1)
        Vi = (mean_vis * (1.0 - mean_vis) * 4.0).clamp(0.0, 1.0)        # 归一化到 [0,1]
        return Vi


class GeometryAwareDecouplingRouter(nn.Module):
    """几何感知解耦路由网络 (GADR)。
    
    输入 3 维特征 [S_hat_i, Ui, Vi]，通道注意力加权后，
    预测对物理先验的残差修正 δ。
    
    alpha_i = Sigmoid(prior_logit + delta)
    
    近零初始化确保初始 delta ≈ 0 → alpha ≈ Sigmoid(prior)。
    
    参数量: 注意力(32+27=59) + 残差网络(64+136+9=209) = 268
    """

    def __init__(self):
        super().__init__()
        # 通道注意力：学习 Si/Ui/Vi 的相对重要性
        self.channel_attention = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),
            nn.Sigmoid()
        )
        # 残差预测网络
        self.residual_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)   # raw delta，不加 Sigmoid
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.channel_attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # 残差分支极小初始化 → 初始 delta ≈ 0
        for m in self.residual_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, features, prior_logit):
        """
        Args:
            features: (N, 3) [S_hat_i, Ui, Vi]
            prior_logit: (N, 1) heuristic prior in logit space

        Returns:
            alpha: (N, 1) in [0, 1]
            delta: (N, 1) residual for monitoring
        """
        att_weights = self.channel_attention(features)    # (N, 3)
        attended = features * att_weights                 # (N, 3)
        delta = self.residual_net(attended)               # (N, 1)
        alpha = torch.sigmoid(prior_logit + delta)        # (N, 1)
        return alpha, delta
