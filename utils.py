import numpy as np
from scipy.interpolate import interp1d
import torch

def kabsch_align(P, Q):
    """
    Align structure P to structure Q using Kabsch algorithm.
    
    Args:
        P: (N,3) array of points to be aligned
        Q: (N,3) array of target points
        
    Returns:
        P_aligned: aligned version of P
        R: optimal rotation matrix
    """
    # Center the points
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Ensure right-handed coordinate system
    d = np.linalg.det(Vt.T @ U.T)
    V = Vt.T
    if d < 0:
        V[:, -1] *= -1
    
    # Compute optimal rotation matrix
    R = V @ U.T
    
    # Apply rotation and translation
    P_aligned = (R @ P.T).T + Q_mean - (R @ P_mean)
    
    return P_aligned, R

def calculate_dihedral(coords):
    # Unpack coordinates into points p1, p2, p3, and p4 for all samples
    p1, p2, p3, p4 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    
    # Calculate bond vectors for each set of points
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Calculate normals to the planes formed by b1-b2 and b2-b3 for each data point
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Calculate m1 for each data point (m1 is perpendicular to n1 and b2)
    m1 = np.cross(n1, b2)
    
    # Calculate dot products for x and y in batch mode
    x = np.einsum('ij,ij->i', n1, n2)    # Batch-wise dot product
    y = np.einsum('ij,ij->i', m1, n2)    # Batch-wise dot product
    
    # Return arctan2 of each set of (y, x) to get dihedral angles in radians
    return -np.arctan2(y, x)

def p_sample(model, x, t):
    batch_size = x.shape[0]
    t = torch.tensor([t]).to(x.device)
    t = t.expand(batch_size)
    
    eps_factor = ((1 - model.extract(model.alphas, t, x)) / model.extract(model.one_minus_alphas_bar_sqrt, t, x))
    eps_theta = model(x, t)
    mean = (1 / model.extract(model.alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    z = torch.randn_like(x)
    sigma_t = model.extract(model.betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, shape):
    cur_x = torch.randn(shape)  # [B, 5, 3]
    x_seq = [cur_x]
    for i in reversed(range(model.n_steps)):
        cur_x = p_sample(model, cur_x, i)
        x_seq.append(cur_x)
    return x_seq

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = model.extract(model.alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = model.extract(model.one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

def noise_estimation_loss(model, x_0):
    batch_size = x_0.shape[0]

    # Select a random step for each example
    t = torch.randint(0, model.n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, model.n_steps - t - 1], dim=0)[:batch_size].long()

    # x0 multiplier
    a = model.extract(model.alphas_bar_sqrt, t, x_0)

    # eps multiplier
    am1 = model.extract(model.one_minus_alphas_bar_sqrt, t, x_0)

    # Generate random noise
    e = torch.randn_like(x_0)

    # model input
    x = x_0 * a + e * am1

    # Get model prediction
    output = model(x, t)

    # Calculate loss
    return (e - output).square().mean()

class EMA(object):
    def __init__(self, mu=0.99):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def filter_by_density(phi, psi, H, xedges, yedges):
    """Filter data points based on their histogram bin density"""

    # Calculate average histogram value
    avg_H = np.mean(H)

    # Find which bin each point belongs to
    phi_indices = np.digitize(phi, xedges) - 1
    psi_indices = np.digitize(psi, yedges) - 1

    # Clip indices to avoid out of bounds
    phi_indices = np.clip(phi_indices, 0, len(xedges)-2)
    psi_indices = np.clip(psi_indices, 0, len(yedges)-2)

    # Get histogram value for each point
    point_densities = H[phi_indices, psi_indices]

    # Create mask for points above average
    mask = point_densities > avg_H

    # Filter points
    phi_filtered = phi[mask]
    psi_filtered = psi[mask]

    print(f"Kept {len(phi_filtered)} points out of {len(phi)} ({100*len(phi_filtered)/len(phi):.1f}%)")

    # Plot original and filtered data
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original data
    #ax1.contour(xedges[:-1], yedges[:-1], -np.log(H.T)+np.log(200000), levels=30, cmap='viridis')
    #ax1.set_xlabel('Phi')
    #ax1.set_ylabel('Psi')
    #ax1.set_title('Original Distribution')

    # Calculate new histogram for filtered data
    #H_filtered, _, _ = np.histogram2d(phi_filtered, psi_filtered, 
    #                                bins=100, 
    #                                range=[[-np.pi, np.pi], [-np.pi, np.pi]])

    # Filtered data
    #ax2.contour(xedges[:-1], yedges[:-1], -np.log(H_filtered.T+1)+np.log(200000), 
    #            levels=30, cmap='viridis')
    #ax2.set_xlabel('Phi')
    #ax2.set_ylabel('Psi')
    #ax2.set_title('Filtered Distribution (Above Average Density)')

    #plt.tight_layout()
    #plt.show()

    return phi_filtered, psi_filtered, mask

