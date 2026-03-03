import torch
import torch.nn as nn
import torch.nn.functional as F

class LogiPARLoss(nn.Module):
    def __init__(self, lambda_f=1.0, lambda_s=0.01):
        super().__init__()
        self.lambda_f = lambda_f
        self.lambda_s = lambda_s
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, P, y_true, c, c_true, reasoning_module):
        # P: [Y, B], y_true: [B], c: [B, N], c_true: [B, N] if available
        # reasoning_module: to get sparsity

        # Cross-entropy: note P is [Y, B], y_true [B]
        l_ce = self.ce_loss(P.T, y_true)

        # Fact grounding: BCE between c and c_true
        l_fact = self.bce_loss(c, c_true) if c_true is not None else 0

        # Sparsity
        l_sparse = reasoning_module.get_sparsity_loss()

        total_loss = l_ce + self.lambda_f * l_fact + self.lambda_s * l_sparse

        return total_loss, l_ce, l_fact, l_sparse