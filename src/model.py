"""
model.py

Defines the Multi-View Linear Manifold (MVLM) architecture and the 
Hybrid Geometric Loss function for drug-target interaction prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewLinearManifold(nn.Module):
    """
    Projects pre-trained molecular and proteomic embeddings into a shared 
    Euclidean manifold space using regularized linear transformations.
    """
    def __init__(self, drug_dim=768, protein_dim=480, latent_dim=512):
        super().__init__()
        # Linear projections with spectral normalization to enforce Lipschitz continuity
        self.U = nn.utils.spectral_norm(nn.Linear(drug_dim, latent_dim, bias=False))
        self.V = nn.utils.spectral_norm(nn.Linear(protein_dim, latent_dim, bias=False))
        
        # Layer normalization for geometric stability
        self.ln_d = nn.LayerNorm(latent_dim)
        self.ln_p = nn.LayerNorm(latent_dim)
        
        # Learnable temperature scalar for sigmoid conversion
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, d_emb, p_emb):
        """
        Computes the geometric alignment (dot product of L2-normalized vectors)
        between drug and protein embeddings in the shared manifold.
        """
        d_proj = self.ln_d(self.U(d_emb))
        p_proj = self.ln_p(self.V(p_emb))
        
        # L2 normalization projects vectors onto a unit hypersphere
        d_norm = F.normalize(d_proj, dim=-1)
        p_norm = F.normalize(p_proj, dim=-1)
        
        # Calculate cosine similarity (equivalent to dot product of normalized vectors)
        similarity = torch.sum(d_norm * p_norm, dim=1)
        return similarity


class HybridLoss(nn.Module):
    """
    A hybrid objective combining Weighted Binary Cross-Entropy (WBCE) 
    and a geometric contrastive loss (Mean Squared Error on cosine similarities).
    """
    def __init__(self, alpha=0.3, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, scores, targets, pos_weight=None):
        """
        Args:
            scores: Raw cosine similarities from the MVLM forward pass.
            targets: Binary interaction labels (0 or 1).
            pos_weight: Dynamic weight for the positive class to handle imbalance.
        """
        # Temperature scaling for BCE logits
        logits = scores / self.temperature
        
        loss_bce = self.bce(logits, targets)
        if pos_weight is not None:
            weight_vector = torch.where(targets == 1.0, pos_weight, torch.tensor(1.0).to(targets.device))
            loss_bce = loss_bce * weight_vector
        loss_bce = loss_bce.mean()
        
        # Geometric constraint: spatial orthogonalization for inactive pairs
        loss_geo = torch.mean((scores - targets) ** 2) 

        return (1 - self.alpha) * loss_bce + self.alpha * loss_geo
