import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareFusion(nn.Module):
    def __init__(self, num_facts=24, num_views=4, epsilon=1e-8):
        super().__init__()
        self.num_facts = num_facts
        self.num_views = num_views
        self.epsilon = epsilon

    def forward(self, z, rho):
        # z: [V, B, N], rho: [V, B, N]
        V, B, N = z.shape

        # Compute view attribution weights v_k^{(v)}
        v = torch.zeros_like(rho)  # [V, B, N]
        for k in range(N):
            for b in range(B):
                rel_scores = rho[:, b, k]  # [V]
                v[:, b, k] = F.softmax(rel_scores, dim=0)  # Normalize over views

        # Fuse logits
        c_logits = torch.zeros(B, N)  # [B, N]
        for k in range(N):
            weighted_z = v[:, :, k] * z[:, :, k]  # [V, B]
            c_logits[:, k] = weighted_z.sum(dim=0)  # [B]

        c = torch.sigmoid(c_logits)  # [B, N]

        return c, v  # c: [B, N], v: [V, B, N]