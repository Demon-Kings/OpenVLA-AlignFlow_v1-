"""
Embodiment Conditioning & Feature-wise Linear Modulation (FiLM) Layers
"""
import torch
import torch.nn as nn
from typing import Optional


class EmbodimentEmbedding(nn.Module):
    """
    Learned Multi-Embodiment Embedding Module:
    Maps robot embodiment ID (0: WidowX, 1: Google Robot, 2: Franka Panda) to dense continuous vector.
    """

    def __init__(self, num_embodiments: int = 3, embed_dim: int = 128):
        super().__init__()
        self.num_embodiments = num_embodiments
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embodiments, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, embodiment_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embodiment_id: (B,) long tensor or int
        Returns:
            emb_vec: (B, embed_dim) float tensor
        """
        if not isinstance(embodiment_id, torch.Tensor):
            embodiment_id = torch.tensor(embodiment_id, dtype=torch.long, device=self.embedding.weight.device)
        else:
            if embodiment_id.dtype != torch.long:
                embodiment_id = embodiment_id.long()
            if embodiment_id.device != self.embedding.weight.device:
                embodiment_id = embodiment_id.to(self.embedding.weight.device)

        if embodiment_id.ndim == 0:
            embodiment_id = embodiment_id.unsqueeze(0)
        raw_emb = self.embedding(embodiment_id)
        return self.proj(raw_emb)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM):
    Applies affine transformation to intermediate activations h conditioned on embodiment vector e:
    FiLM(h, e) = (1 + gamma(e)) * h + beta(e)
    """

    def __init__(self, condition_dim: int, target_dim: int):
        super().__init__()
        self.fc_gamma = nn.Linear(condition_dim, target_dim)
        self.fc_beta = nn.Linear(condition_dim, target_dim)

        # Initialize to identity transform (gamma=0, beta=0)
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, ..., target_dim)
            condition: (B, condition_dim)
        Returns:
            modulated_h: (B, ..., target_dim)
        """
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)

        # Reshape for broadcasting if h has more than 2 dimensions
        if h.ndim > 2:
            dims_to_add = h.ndim - condition.ndim
            for _ in range(dims_to_add):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        gamma = torch.clamp(gamma, min=-2.0, max=2.0)
        beta = torch.clamp(beta, min=-5.0, max=5.0)
        out = (1.0 + gamma) * h + beta
        return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
