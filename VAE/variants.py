# VAE/variants.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from VAE.network import VAE

#
# --- VectorQuantizer (self-contained) ---
#
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_embeddings  = num_embeddings
        self.commitment_cost = commitment_cost

        # codebook embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1.0/self.num_embeddings, 1.0/self.num_embeddings
        )

    def forward(self, inputs):
        # inputs: (batch, embedding_dim)
        flat = inputs.view(-1, self.embedding_dim)  # (B', D)
        # compute distances
        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * flat @ self.embeddings.weight.t()
        )  # (B', N)
        # encoding
        indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B',1)
        encodings = torch.zeros(indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, indices, 1)
        # quantize
        quantized = (encodings @ self.embeddings.weight).view_as(inputs)
        # losses
        q_loss = F.mse_loss(quantized.detach(), inputs)
        e_loss = F.mse_loss(quantized, inputs.detach())
        loss   = q_loss + self.commitment_cost * e_loss
        # straight-through
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

#
# --- Maximum Mean Discrepancy (MMD) kernel for WAE ---
#
def compute_mmd(z, prior_z, sigma=1.0):
    z2  = torch.cdist(z,      z,      p=2).pow(2)
    p2  = torch.cdist(prior_z, prior_z, p=2).pow(2)
    zp2 = torch.cdist(z,      prior_z, p=2).pow(2)
    def rbf(d2): return torch.exp(-d2 / (2 * sigma * sigma))
    return rbf(z2).mean() + rbf(p2).mean() - 2 * rbf(zp2).mean()

#
# --- WAE (Wasserstein Auto-Encoder) ---
#
class WAE(VAE):
    def __init__(self, x_dim, h_dim, z_dim, lambda_mmd=10.0, sigma=1.0):
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.lambda_mmd = lambda_mmd
        self.sigma      = sigma

    def loss_function(self, x_recon, x, mu, **kwargs):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        prior = torch.randn_like(mu)
        mmd   = compute_mmd(mu, prior, self.sigma)
        total = recon + self.lambda_mmd * mmd
        return total, {'recon': recon, 'mmd': mmd}

#
# --- RAE (Regularized Auto-Encoder) ---
#
class RAE(VAE):
    def __init__(self, x_dim, h_dim, z_dim, lambda_z=0.1):
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.lambda_z = lambda_z

    def loss_function(self, x_recon, x, mu, **kwargs):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        zreg  = mu.pow(2).sum()
        total = recon + self.lambda_z * zreg
        return total, {'recon': recon, 'zreg': zreg}

#
# --- VQ-VAE (Vector Quantized VAE) ---
#
class VQVAE(VAE):
    def __init__(self, x_dim, h_dim, z_dim, num_embeddings=512, commitment_cost=0.25):
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.codebook = VectorQuantizer(num_embeddings, z_dim, commitment_cost)

    def forward(self, x):
        # encode returns (mu, logvar)
        mu, logvar = self.encode(x)
        # quantize the mu vectors
        z_q, vq_loss = self.codebook(mu)
        # decode the quantized vectors
        x_recon = self.decode(z_q)
        return x_recon, mu, z_q, vq_loss

    def loss_function(self, x_recon, x, mu, z_q, vq_loss):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        total = recon + vq_loss
        return total, {'recon': recon, 'vq_loss': vq_loss}
