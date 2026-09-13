import torch
import torch.nn as nn
import torchvision.models as models
import math
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np  # Assuming data loading uses NumPy or similar

# Utility functions for SE(3) operations (ported from JAX to PyTorch)

# https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

# https://arwilliams.github.io/so3-exp.pdf
# Convert 3D vector to skew-symmetric matrix
def hat(w):
    """
    Convert 3D vector(s) to skew-symmetric matrix/matrices.
    
    Args:
        w: Tensor of shape (3,) or (..., 3) - 3D vector(s)
        
    Returns:
        Tensor of shape (3, 3) or (..., 3, 3) - skew-symmetric matrix/matrices
    """
    original_shape = w.shape
    if len(original_shape) == 1:  # Single vector (3,)
        device = w.device
        zero = torch.zeros_like(w[0])
        return torch.stack([
            torch.stack([zero, -w[2], w[1]]),
            torch.stack([w[2], zero, -w[0]]),
            torch.stack([-w[1], w[0], zero])
        ]).to(device)
    else:  # Batch processing (..., 3)
        # Flatten to (N, 3) for processing
        w_flat = w.view(-1, 3)
        N = w_flat.shape[0]
        
        # Vectorized skew-symmetric matrix creation
        zero = torch.zeros(N, device=w.device, dtype=w.dtype)
        K = torch.zeros(N, 3, 3, device=w.device, dtype=w.dtype)
        
        # Fill the skew-symmetric matrix elements
        K[:, 0, 1] = -w_flat[:, 2]
        K[:, 0, 2] = w_flat[:, 1]
        K[:, 1, 0] = w_flat[:, 2]
        K[:, 1, 2] = -w_flat[:, 0]
        K[:, 2, 0] = -w_flat[:, 1]
        K[:, 2, 1] = w_flat[:, 0]
        
        # Reshape back to original batch shape + (3, 3)
        return K.view(*original_shape[:-1], 3, 3)

# Convert skew-symmetric matrix to 3D vector
def vee(K):
    return torch.stack([K[2,1], K[0,2], K[1,0]])

def exp_so3(omega):
    theta = torch.norm(omega)
    if theta < 1e-6:
        return torch.eye(3, device=omega.device)
    w = omega / theta
    K = hat(w)
    K2 = K @ K
    R = torch.eye(3, device=omega.device) + math.sin(theta) * K + (1 - math.cos(theta)) * K2
    return R

# https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf
def log_so3(R):
    # Handle different input shapes: (3, 3), (batch, 3, 3), or (batch, timesteps, 3, 3)
    original_shape = R.shape
    
    if len(original_shape) == 2:  # Single matrix (3, 3)
        tr = torch.trace(R)
        if tr > 3 - 1e-6:
            theta = 0
        else:
            theta = torch.acos(torch.clamp((tr - 1) / 2, -1.0, 1.0))
        if theta < 1e-6:
            return torch.zeros(3, device=R.device)
        lnR = theta / (2 * torch.sin(theta)) * (R - R.t())
        omega = vee(lnR)
        return omega
    
    else:  # Vectorized batch processing for (batch, 3, 3) or (batch, timesteps, 3, 3)
        # Flatten to (N, 3, 3) for vectorized processing
        original_batch_shape = original_shape[:-2]
        R_flat = R.view(-1, 3, 3)
        N = R_flat.shape[0]
        
        # Vectorized trace computation
        tr = torch.diagonal(R_flat, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        
        # Vectorized theta computation
        theta = torch.where(
            tr > 3 - 1e-6,
            torch.zeros_like(tr),
            torch.acos(torch.clamp((tr - 1) / 2, -1.0, 1.0))
        )
        
        # Initialize omega
        omega = torch.zeros((N, 3), device=R.device)
        
        # Find non-small theta indices
        valid_mask = theta >= 1e-6
        
        if valid_mask.any():
            R_valid = R_flat[valid_mask]
            theta_valid = theta[valid_mask]
            
            # Vectorized computation for valid cases
            sin_theta = torch.sin(theta_valid)
            coeff = theta_valid / (2 * sin_theta)
            
            # R - R.transpose for valid matrices
            skew_matrices = R_valid - R_valid.transpose(-2, -1)
            
            # Extract omega using vectorized vee operation
            omega[valid_mask, 0] = coeff * skew_matrices[:, 2, 1]
            omega[valid_mask, 1] = coeff * skew_matrices[:, 0, 2]
            omega[valid_mask, 2] = coeff * skew_matrices[:, 1, 0]
        
        # Reshape back to original batch shape
        return omega.view(*original_batch_shape, 3)

def exp_se3(xi):
    # Handle different input shapes: (6,), (batch, 6), or (batch, timesteps, 6)
    original_shape = xi.shape
    
    if len(original_shape) == 1:  # Single vector (6,)
        omega = xi[:3]
        v = xi[3:]
        
        theta = torch.norm(omega)
        if theta < 1e-6:
            R = torch.eye(3, device=xi.device)
            V = torch.eye(3, device=xi.device)
        else:
            w = omega / theta
            K = hat(w)
            K2 = K @ K
            R = torch.eye(3, device=xi.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K2
            V = torch.eye(3, device=xi.device) + (1 - torch.cos(theta)) / theta * K + (theta - torch.sin(theta)) / (theta ** 2) * K2
        translation = V @ v
        T = torch.eye(4, device=xi.device)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T
    
    else:  # Vectorized batch processing for (batch, 6) or (batch, timesteps, 6)
        # Flatten to (N, 6) for vectorized processing
        original_batch_shape = original_shape[:-1]
        xi_flat = xi.view(-1, 6)
        N = xi_flat.shape[0]
        
        omega = xi_flat[:, :3]  # (N, 3)
        v = xi_flat[:, 3:]      # (N, 3)
        
        # Vectorized theta computation
        theta = torch.norm(omega, dim=-1)  # (N,)
        
        # Initialize outputs
        T = torch.eye(4, device=xi.device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
        
        # Small theta case (identity rotation)
        small_theta_mask = theta < 1e-6
        
        # Large theta case
        large_theta_mask = ~small_theta_mask
        
        if large_theta_mask.any():
            omega_large = omega[large_theta_mask]  # (M, 3)
            v_large = v[large_theta_mask]          # (M, 3)
            theta_large = theta[large_theta_mask]  # (M,)
            
            # Vectorized hat operation for multiple vectors
            w = omega_large / theta_large.unsqueeze(-1)  # (M, 3)
            
            # Vectorized skew-symmetric matrices
            zero = torch.zeros_like(w[:, 0])
            K = torch.stack([
                torch.stack([zero, -w[:, 2], w[:, 1]], dim=-1),
                torch.stack([w[:, 2], zero, -w[:, 0]], dim=-1),
                torch.stack([-w[:, 1], w[:, 0], zero], dim=-1)
            ], dim=-2)  # (M, 3, 3)
            
            K2 = K @ K  # (M, 3, 3)
            
            # Vectorized Rodrigues formula
            I = torch.eye(3, device=xi.device).unsqueeze(0).repeat(K.shape[0], 1, 1)
            sin_theta = torch.sin(theta_large).unsqueeze(-1).unsqueeze(-1)
            cos_theta = torch.cos(theta_large).unsqueeze(-1).unsqueeze(-1)
            
            R = I + sin_theta * K + (1 - cos_theta) * K2  # (M, 3, 3)
            
            # Vectorized V matrix computation
            theta_large_expanded = theta_large.unsqueeze(-1).unsqueeze(-1)
            V = (I + 
                 (1 - cos_theta) / theta_large_expanded * K + 
                 (theta_large_expanded - sin_theta) / (theta_large_expanded ** 2) * K2)
            
            # Vectorized translation
            translation = torch.bmm(V, v_large.unsqueeze(-1)).squeeze(-1)  # (M, 3)
            
            # Fill in the transformation matrices
            T[large_theta_mask, :3, :3] = R
            T[large_theta_mask, :3, 3] = translation
        
        # Small theta case - already identity for rotation, just copy translation
        if small_theta_mask.any():
            T[small_theta_mask, :3, 3] = v[small_theta_mask]
        
        # Reshape back to original batch shape
        return T.view(*original_batch_shape, 4, 4)

def log_se3(T):
    # Handle different input shapes: (4, 4), (batch, 4, 4), or (batch, timesteps, 4, 4)
    original_shape = T.shape
    
    if len(original_shape) == 2:  # Single matrix (4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        omega = log_so3(R)
        theta = torch.norm(omega)
        if theta < 1e-6:
            Vinv = torch.eye(3, device=T.device)
        else:
            w = omega / theta
            K = hat(w)
            K2 = K @ K
            Vinv = torch.eye(3, device=T.device) - 0.5 * K + (1 - theta / (2 * torch.tan(theta / 2))) / (theta ** 2) * K2
        v = Vinv @ t
        xi = torch.cat([omega, v])
        return xi
    
    else:  # Vectorized batch processing for (batch, 4, 4) or (batch, timesteps, 4, 4)
        # Flatten to (N, 4, 4) for vectorized processing
        original_batch_shape = original_shape[:-2]
        T_flat = T.view(-1, 4, 4)
        N = T_flat.shape[0]
        
        # Extract rotation matrices and translations (vectorized)
        R = T_flat[:, :3, :3]  # (N, 3, 3)
        t = T_flat[:, :3, 3]   # (N, 3)
        
        # Vectorized log_so3 computation
        omega = log_so3(R)  # (N, 3)
        theta = torch.norm(omega, dim=-1)  # (N,)
        
        # Initialize Vinv matrices
        I = torch.eye(3, device=T.device).unsqueeze(0).repeat(N, 1, 1)  # (N, 3, 3)
        Vinv = I.clone()
        
        # Find non-small theta indices
        valid_mask = theta >= 1e-6
        
        if valid_mask.any():
            omega_valid = omega[valid_mask]  # (M, 3)
            theta_valid = theta[valid_mask]  # (M,)
            
            # Vectorized computation for valid cases
            w = omega_valid / theta_valid.unsqueeze(-1)  # (M, 3)
            
            # Vectorized skew-symmetric matrices
            zero = torch.zeros_like(w[:, 0])
            K = torch.stack([
                torch.stack([zero, -w[:, 2], w[:, 1]], dim=-1),
                torch.stack([w[:, 2], zero, -w[:, 0]], dim=-1),
                torch.stack([-w[:, 1], w[:, 0], zero], dim=-1)
            ], dim=-2)  # (M, 3, 3)
            
            K2 = K @ K  # (M, 3, 3)
            
            # Vectorized Vinv computation
            I_valid = I[valid_mask]  # (M, 3, 3)
            theta_expanded = theta_valid.unsqueeze(-1).unsqueeze(-1)  # (M, 1, 1)
            tan_half_theta = torch.tan(theta_valid / 2).unsqueeze(-1).unsqueeze(-1)  # (M, 1, 1)
            
            Vinv_valid = (I_valid - 0.5 * K + 
                         (1 - theta_expanded / (2 * tan_half_theta)) / (theta_expanded ** 2) * K2)
            
            Vinv[valid_mask] = Vinv_valid
        
        # Vectorized translation transformation
        v = torch.bmm(Vinv, t.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        
        # Combine omega and v
        xi = torch.cat([omega, v], dim=-1)  # (N, 6)
        
        # Reshape back to original batch shape
        return xi.view(*original_batch_shape, 6)

def compose_se3(T1, T2):
    return T1 @ T2

def se3_to_trans_quat(se3_matrices):
    """
    Convert SE(3) transformation matrices to translation + quaternion representation.
    
    Args:
        se3_matrices: Tensor of shape (..., 4, 4) - SE(3) transformation matrices
        
    Returns:
        Tensor of shape (..., 7) where the last dimension is [x, y, z, qw, qx, qy, qz]
    """
    # Handle different input shapes
    original_shape = se3_matrices.shape[:-2]  # All dimensions except last two (4, 4)
    
    # Flatten to (N, 4, 4) for vectorized processing
    se3_flat = se3_matrices.view(-1, 4, 4)
    N = se3_flat.shape[0]
    
    # Extract translation (vectorized)
    translation = se3_flat[:, :3, 3]  # (N, 3)
    
    # Extract rotation matrices (vectorized)
    rotation_matrices = se3_flat[:, :3, :3]  # (N, 3, 3)
    
    # Convert rotation matrices to quaternions (vectorized)
    quaternions = mat_to_quat(rotation_matrices)  # (N, 4) -> [qw, qx, qy, qz]
    
    # Combine translation and quaternion
    trans_quat = torch.cat([translation, quaternions], dim=-1)  # (N, 7)
    
    # Reshape back to original batch shape + 7
    return trans_quat.view(*original_shape, 7)

def mat_to_quat(rotation_matrices):
    """
    Convert rotation matrices to quaternions using Shepperd's method (vectorized).
    
    Args:
        rotation_matrices: Tensor of shape (..., 3, 3) - rotation matrices
        
    Returns:
        Tensor of shape (..., 4) where the last dimension is [qw, qx, qy, qz]
    """
    # Handle different input shapes
    original_shape = rotation_matrices.shape[:-2]  # All dimensions except last two (3, 3)
    
    if len(original_shape) == 0:  # Single matrix (3, 3)
        R = rotation_matrices
        
        # Shepperd's method for single matrix
        trace = torch.trace(R)
        
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return torch.stack([qw, qx, qy, qz], dim=0)
    
    else:  # Vectorized batch processing
        # Flatten to (N, 3, 3) for vectorized processing
        R_flat = rotation_matrices.view(-1, 3, 3)
        N = R_flat.shape[0]
        
        # Compute traces for all matrices
        traces = torch.diagonal(R_flat, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        
        # Initialize quaternions
        quats = torch.zeros(N, 4, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
        
        # Case 1: trace > 0
        mask1 = traces > 0
        if mask1.any():
            R1 = R_flat[mask1]  # (M1, 3, 3)
            trace1 = traces[mask1]  # (M1,)
            s = torch.sqrt(trace1 + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R1[:, 2, 1] - R1[:, 1, 2]) / s
            qy = (R1[:, 0, 2] - R1[:, 2, 0]) / s
            qz = (R1[:, 1, 0] - R1[:, 0, 1]) / s
            quats[mask1] = torch.stack([qw, qx, qy, qz], dim=-1)
        
        # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
        remaining_mask = ~mask1
        if remaining_mask.any():
            R_rem = R_flat[remaining_mask]
            mask2 = (R_rem[:, 0, 0] > R_rem[:, 1, 1]) & (R_rem[:, 0, 0] > R_rem[:, 2, 2])
            
            if mask2.any():
                R2 = R_rem[mask2]  # (M2, 3, 3)
                s = torch.sqrt(1.0 + R2[:, 0, 0] - R2[:, 1, 1] - R2[:, 2, 2]) * 2  # s = 4 * qx
                qw = (R2[:, 2, 1] - R2[:, 1, 2]) / s
                qx = 0.25 * s
                qy = (R2[:, 0, 1] + R2[:, 1, 0]) / s
                qz = (R2[:, 0, 2] + R2[:, 2, 0]) / s
                
                # Map back to original indices
                remaining_indices = torch.where(remaining_mask)[0]
                mask2_indices = remaining_indices[mask2]
                quats[mask2_indices] = torch.stack([qw, qx, qy, qz], dim=-1)
            
            # Case 3: R[1,1] > R[2,2]
            mask3 = ~mask2 & (R_rem[:, 1, 1] > R_rem[:, 2, 2])
            if mask3.any():
                R3 = R_rem[mask3]  # (M3, 3, 3)
                s = torch.sqrt(1.0 + R3[:, 1, 1] - R3[:, 0, 0] - R3[:, 2, 2]) * 2  # s = 4 * qy
                qw = (R3[:, 0, 2] - R3[:, 2, 0]) / s
                qx = (R3[:, 0, 1] + R3[:, 1, 0]) / s
                qy = 0.25 * s
                qz = (R3[:, 1, 2] + R3[:, 2, 1]) / s
                
                # Map back to original indices
                remaining_indices = torch.where(remaining_mask)[0]
                mask3_indices = remaining_indices[mask3]
                quats[mask3_indices] = torch.stack([qw, qx, qy, qz], dim=-1)
            
            # Case 4: else (R[2,2] is largest)
            mask4 = ~mask2 & ~mask3
            if mask4.any():
                R4 = R_rem[mask4]  # (M4, 3, 3)
                s = torch.sqrt(1.0 + R4[:, 2, 2] - R4[:, 0, 0] - R4[:, 1, 1]) * 2  # s = 4 * qz
                qw = (R4[:, 1, 0] - R4[:, 0, 1]) / s
                qx = (R4[:, 0, 2] + R4[:, 2, 0]) / s
                qy = (R4[:, 1, 2] + R4[:, 2, 1]) / s
                qz = 0.25 * s
                
                # Map back to original indices
                remaining_indices = torch.where(remaining_mask)[0]
                mask4_indices = remaining_indices[mask4]
                quats[mask4_indices] = torch.stack([qw, qx, qy, qz], dim=-1)
        
        # Reshape back to original batch shape + 4
        return quats.view(*original_shape, 4)

def quat_to_mat(q):
    # q shape: (batch, timesteps, 4) where each quaternion is wxyz
    # Returns rotation matrices with shape (batch, timesteps, 3, 3)
    
    # Handle various input shapes
    original_shape = q.shape
    if len(original_shape) == 1:  # Single quaternion (4,)
        q = q.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
    elif len(original_shape) == 2:  # (batch, 4) or (timesteps, 4)
        q = q.unsqueeze(1)  # (batch, 1, 4) or (timesteps, 1, 4)
    
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute rotation matrix elements
    R00 = 1 - 2*y**2 - 2*z**2
    R01 = 2*x*y - 2*z*w
    R02 = 2*x*z + 2*y*w
    
    R10 = 2*x*y + 2*z*w
    R11 = 1 - 2*x**2 - 2*z**2
    R12 = 2*y*z - 2*x*w
    
    R20 = 2*x*z - 2*y*w
    R21 = 2*y*z + 2*x*w
    R22 = 1 - 2*x**2 - 2*y**2
    
    # Stack to form rotation matrices (batch, timesteps, 3, 3)
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1)
    ], dim=-2)
    
    # Restore original batch/timestep structure
    if len(original_shape) == 1:
        R = R.squeeze(0).squeeze(0)  # (3, 3)
    elif len(original_shape) == 2:
        R = R.squeeze(1)  # (batch, 3, 3) or (timesteps, 3, 3)
    
    return R

class PowerNoiseSchedule:
    def __init__(self, alpha_start=1e-8, alpha_end=1.0, timesteps=500, power=3.0):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.power = power
        self.timesteps = timesteps
        self.alphas = self.create_alphas(timesteps)
        self.bar_alpha_t = torch.cumprod(torch.cat((torch.tensor([1.0]), self.alphas))[:-1], dim=0)
        self.sqrt_alphas = torch.sqrt(self.bar_alpha_t)
        self.sigmas = torch.sqrt(1.0 - self.bar_alpha_t)

    def create_alphas(self, timesteps):
        return (
            torch.linspace(
                self.alpha_start ** (1 / self.power),
                self.alpha_end ** (1 / self.power),
                timesteps,
            )
            ** self.power
        )

    def get_sigma(self, t):
        return self.sigmas[t]

# Sinusoidal Positional Encoding
def sinusoidal_positional_encoding(x, num_levels):
    half_dim = num_levels
    freq = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -math.log(10000.0) / half_dim).to(x.device)
    angles = x.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb

# Fourier-Conditioned Linear Layer (from Eq. 13 in the paper)
class FourierConditionedLinear(nn.Module):
    def __init__(self, in_features, out_features, cond_dim):
        super().__init__()
        self.A = nn.Linear(cond_dim, in_features)
        self.B = nn.Linear(cond_dim, in_features)
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x, c):
        a = self.A(c)
        b = self.B(c)
        transformed = a * torch.cos(math.pi * x) + b * torch.sin(math.pi * x)
        return self.W(transformed)

# Score Network Module (from Section 4.4 in the paper)
class ScoreNet(nn.Module):
    def __init__(self, pose_dim=6, image_channels=3, cond_dim=640, hidden_dim=512, num_layers=8, pos_emb_levels=16):
        super().__init__()
        # Image encoder (ResNet18, output to 512-dim feature)
        self.image_encoder = models.resnet18(pretrained=False)
        self.image_encoder.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)

        # Time embedding (assuming max timesteps=1000, output 128-dim)
        self.time_emb = nn.Embedding(1000, 128)

        # Positional encoding for noisy pose (input dim after embedding: pose_dim * 2 * pos_emb_levels)
        self.pos_emb_levels = pos_emb_levels
        input_dim = pose_dim * 2 * pos_emb_levels

        # Conditioning dim (image_feat 512 + time_emb 128 = 640)
        self.cond_dim = cond_dim

        # MLP blocks with Fourier conditioning
        self.layers = nn.ModuleList()
        in_d = input_dim
        for _ in range(num_layers):
            self.layers.append(FourierConditionedLinear(in_d, hidden_dim, cond_dim))
            self.layers.append(nn.SiLU())  # Assuming SiLU as in many diffusion models
            in_d = hidden_dim
        self.layers.append(FourierConditionedLinear(in_d, pose_dim, cond_dim))  # Output to 6D (tangent space)

    def forward(self, img, rt, t):
        # rt: noisy pose in tangent space (batch_size, 6)
        # Embed noisy pose
        pose_emb = sinusoidal_positional_encoding(rt, self.pos_emb_levels).view(rt.size(0), -1)

        # Image features
        image_feat = self.image_encoder(img)

        # Time embedding
        time_feat = self.time_emb(t)

        # Condition c
        c = torch.cat([image_feat, time_feat], dim=-1)

        # Forward through MLP
        x = pose_emb
        for layer in self.layers:
            if isinstance(layer, FourierConditionedLinear):
                x = layer(x, c)
            else:
                x = layer(x)
        return x  # Predicted unit noise z (6D)

# Custom Dataset for Score Matching (emulates sample_train_fn)
class PoseDiffusionDataset(Dataset):
    def __init__(self, images, rotations, translations, noise_schedule, repr_type='tan'):
        # images: (N, H, W, C) torch.Tensor or np.array
        # rotations: (N, 4) quaternions (wxyz)
        # translations: (N, 3) xyz
        self.images = torch.tensor(images) if isinstance(images, np.ndarray) else images
        self.rotations = torch.tensor(rotations) if isinstance(rotations, np.ndarray) else rotations
        self.translations = torch.tensor(translations) if isinstance(translations, np.ndarray) else translations
        self.noise_schedule = noise_schedule
        self.repr_type = repr_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        rot = self.rotations[idx]
        tran = self.translations[idx]

        # Sample timestep t (keep as tensor for GPU efficiency)
        t = torch.randint(0, self.noise_schedule.timesteps, (1,), device=rot.device).squeeze()
        sigma = self.noise_schedule.get_sigma(t)

        # Clean pose r0 as 4x4 matrix
        R = quat_to_mat(rot)
        r0 = torch.eye(4, device=rot.device)
        r0[:3, :3] = R
        r0[:3, 3] = tran

        # Sample unit noise z ~ N(0, I) in tangent space (6D)
        z = torch.randn(6, device=rot.device)

        # Perturb: rt = r0 @ exp_se3(sigma * z)
        delta = exp_se3(sigma * z)
        rt = compose_se3(r0, delta)

        # Convert to representation (tangent for input)
        if self.repr_type == 'tan':
            rt_repr = log_se3(rt)
            r0_repr = log_se3(r0)
            zt = z  # Unit z (target for model prediction)
        else:
            raise NotImplementedError("Only 'tan' repr_type supported")

        return {
            'img': img,
            'rt': rt_repr,
            't': t,  # Keep as tensor
            'zt': zt,
            'r0': r0_repr
        }

# Training step for Denoising Score Matching (DSM)
def train_step(model, optimizer, batch, learn_noise=True, loss_name='euclidean'):
    img = batch['img']
    rt = batch['rt']
    t = batch['t']
    target = batch['zt'] if learn_noise else batch['r0']

    optimizer.zero_grad()
    mu = model(img, rt, t)
    
    # Loss (euclidean distance, i.e., MSE)
    if loss_name == 'euclidean':
        loss = ((mu - target) ** 2).mean()
    else:
        raise NotImplementedError("Only 'euclidean' loss supported")

    loss.backward()
    optimizer.step()
    return loss.item()

def sample_poses(model, img, noise_schedule, num_samples=1, num_steps=None, eps=0.01, num_sub_steps=5, device='cuda'):
    """
    Inference/sampling function for Lie diffusion on SE(3) using score matching.
    This implements the reverse diffusion process (geodesic random walk) from the paper (Eq. 7).
    
    Args:
        model (ScoreNet): Trained score network that predicts the unit noise z.
        img (torch.Tensor): Input image (batch_size, C, H, W) for conditioning.
        noise_schedule (NoiseSchedule): Noise schedule with sigmas.
        num_samples (int): Number of pose samples to generate (for handling ambiguity).
        num_steps (int, optional): Number of denoising steps (defaults to noise_schedule.timesteps).
        eps (float): Step size ϵ_i for the geodesic random walk.
        num_sub_steps (int): Number of small sub-steps per timestep (to approximate the surrogate score better, as per Section 4.3 and Fig. 2).
        device (str): Device to run on.
    
    Returns:
        torch.Tensor: Sampled poses as 4x4 SE(3) matrices (num_samples, 4, 4).
    """
    model.eval()
    if num_steps is None:
        num_steps = noise_schedule.timesteps
    
    # Start from prior: high-noise poses (sampled from max sigma Gaussian in tangent space)
    sigma_max = noise_schedule.sigmas[-1]  # Highest noise level
    z_prior = torch.randn((num_samples, 6), device=device) * sigma_max
    
    # Vectorized exp_se3 for initial poses
    x = exp_se3(z_prior)  # (num_samples, 4, 4) - fully vectorized
    
    # Repeat image for multiple samples if needed (assuming single image conditioning)
    if num_samples > 1:
        img = img.repeat(num_samples, 1, 1, 1)
    
    # Reverse loop over timesteps (from high sigma to low)
    for t in reversed(range(num_steps)):
        sigma_t = noise_schedule.get_sigma(t)
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
        
        # Predict the unit noise z using the model (input: tangent space of current x)
        # Vectorized log_se3 for all samples at once
        rt_tan = log_se3(x)  # (num_samples, 6) - fully vectorized
        mu = model(img, rt_tan, t_tensor)  # Predicted z (num_samples, 6)
        
        # Compute surrogate score: s = -mu / sigma_t^2 (Eq. 12)
        score = -mu / (sigma_t ** 2)
        
        # Apply small sub-steps for better approximation (Section 4.3, Fig. 2 right)
        for _ in range(num_sub_steps):
            # Geodesic random walk update (Eq. 7): Δ = ϵ * s + √(2ϵ) * z_i
            z_i = torch.randn((num_samples, 6), device=device)
            delta_tan = eps * score + math.sqrt(2 * eps) * z_i  # (num_samples, 6)
            
            # Vectorized exponentiate to SE(3) delta matrix
            delta = exp_se3(delta_tan)  # (num_samples, 4, 4) - fully vectorized
            
            # Vectorized update pose: x_{i+1} = x_i ∘ delta
            # Using batch matrix multiplication for composition
            x = torch.bmm(x, delta)  # (num_samples, 4, 4) - fully vectorized
    
    return x  # Final denoised poses as 4x4 matrices

# Example usage
# Assume you have images, rotations, translations loaded as tensors or arrays
# noise_schedule = NoiseSchedule(timesteps=1000)
# dataset = PoseDiffusionDataset(images, rotations, translations, noise_schedule)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = ScoreNet()
# optimizer = Adam(model.parameters(), lr=1e-4)

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         loss = train_step(model, optimizer, batch)
#         print(f"Loss: {loss}")

# Note: During inference/denoising, the predicted mu approximates the unit noise z.
# The score is computed as score = -mu / sigma, where sigma = noise_schedule.get_sigma(t)
# Then use in geodesic random walk: next_pose = current_pose @ exp_se3(eps * score + sqrt(2*eps) * torch.randn(6))