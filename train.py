import torch
import torch.nn as nn
import torch.optim as optim
from utils import p_sample, p_sample_loop, q_sample
import MDAnalysis as mda
import matplotlib.pyplot as plt
from utils import filter_by_density
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import EMA, noise_estimation_loss, calculate_dihedral
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_steps, hidden_dim=512):
        super().__init__()
        
        self.n_steps = n_steps
        self.betas = self.make_beta_schedule(start=1e-5, end=1e-2)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        triu_indices = torch.triu_indices(5, 5, offset=1)
        self.register_buffer('triu_i', triu_indices[0])
        self.register_buffer('triu_j', triu_indices[1])
        
        self.input_net = nn.Sequential(
            nn.Linear(30 + 64, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.middle_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(6)
        ])
        
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 15)
        )
        
    def get_distances(self, x):
        diff_vectors = x.unsqueeze(2) - x.unsqueeze(1)  # [B, 5, 5, 3]
        distances = diff_vectors[:, self.triu_i, self.triu_j]  # [B, 10, 3]
        return distances.reshape(x.shape[0], 30)
    
    def make_beta_schedule(self, start=1e-5, end=1e-2):
        betas = torch.linspace(-6, 6, self.n_steps)
        betas = torch.sigmoid(betas) * (end - start) + start
        return betas
        
    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)
    
    def forward(self, x, t):
        B = x.shape[0]
        
        distances = self.get_distances(x)
        
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
            
        t_emb = self.time_embed(t)
        features = torch.cat([distances, t_emb], dim=1)
        x = self.input_net(features)
        
        for layer in self.middle_net:
            x = x + layer(x)  # Residual connection
            
        output = self.final_net(x)
        
        return output.reshape(B, 5, 3)

# Load the trajectory
universe = mda.Universe("simulation/ala2.pdb", "simulation/md_nosol.xtc")


def process_trajectory(topology_file, trajectory_file):
    u = mda.Universe(topology_file, trajectory_file)

    heavy_atom_ids = [4, 6, 8, 14, 16]
    heavy_atoms = u.atoms[heavy_atom_ids]

    dataset = []
    for ts in u.trajectory[::5]:
        positions = heavy_atoms.positions
        com = positions.mean(axis=0)
        positions -= com  

        dataset.append(positions)

    return np.array(dataset)


topology_file = "simulation/ala2.pdb"
trajectory_file = "simulation/md_nosol.xtc"

data = process_trajectory(topology_file, trajectory_file)


phi_atoms = data[:, :-1]
psi_atoms = data[:, 1:]
phi = calculate_dihedral(phi_atoms)
psi = calculate_dihedral(psi_atoms)

H, xedges, yedges = np.histogram2d(phi, psi, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]])

# Apply filtering
phi_filtered, psi_filtered, mask = filter_by_density(phi, psi, H, xedges, yedges)

# If you need to filter the original data array too
data_filtered = data[mask]


model = MLP(n_steps = 1000)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = torch.tensor(data_filtered).float()
batch_size = 1024
steps_per_epoch = dataset.size(0) // batch_size
print(f"Dataset size: {dataset.size(0)}")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")
total_steps = 1000 * steps_per_epoch
ema = EMA(0.999)
ema.register(model)

# Initialize scheduler with actual total steps
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)


# Training loop
best_loss = float('inf')
for epoch in range(1001):
    permutation = torch.randperm(dataset.size()[0])
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, dataset.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = dataset[indices]
        
        # Compute the loss
        loss = noise_estimation_loss(model, batch_x)
        epoch_loss += loss.item()
        n_batches += 1
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        # Update EMA
        ema.update(model)
        
        # Free up memory
        batch_x = batch_x.cpu()
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / n_batches
    
    # Save checkpoint and visualize every 200 epochs
    if epoch % 200 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, Avg Loss: {avg_epoch_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'ema_state_dict': ema.state_dict(),
        }, f'filtered_epoch_{epoch}.pt')
        
