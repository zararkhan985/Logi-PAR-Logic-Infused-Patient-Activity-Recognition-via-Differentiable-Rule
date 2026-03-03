import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralGuidedRuleLearner(nn.Module):
    """Neuro-symbolic rule learning module with differentiable logic.
    
    Learns logical rules from data using Gumbel-Softmax for differentiable
    approximation of discrete selection and negation operations.
    """
    
    def __init__(self, num_facts=24, num_rules=10, num_literals=3, num_classes=2, tau_gs=1.0):
        super().__init__()
        self.num_facts = num_facts
        self.num_rules = num_rules
        self.num_literals = num_literals
        self.num_classes = num_classes
        self.tau_gs = tau_gs

        # Learnable selection matrices: Gamma_{m,j} [M, L, N]
        self.Gamma = nn.Parameter(torch.randn(num_rules, num_literals, num_facts))

        # Negation gates: eta_{m,j} [M, L]
        self.eta = nn.Parameter(torch.sigmoid(torch.randn(num_rules, num_literals)))

        # Rule weights: w_{y,m} [Y, M]
        self.w = nn.Parameter(torch.randn(num_classes, num_rules))

        # Biases: beta_y [Y]
        self.beta = nn.Parameter(torch.zeros(num_classes))

    def forward(self, c, hard=False):
        """Forward pass through the rule learner.
        
        Args:
            c: [B, N] - Grounded facts from fusion module
            hard: bool - If True, use hard Gumbel-Softmax (discrete)
            
        Returns:
            P: [Y, B] - Class probabilities
            gamma: [M, L, N] - Selection weights (Gumbel-softmax output)
            tau: [M, B] - Rule strengths for each sample
        """
        B = c.shape[0]

        # Gumbel-Softmax for differentiable selection: gamma_{m,j} [M, L, N]
        # Use the reparameterization trick with Gumbel noise
        gamma = F.gumbel_softmax(self.Gamma, tau=self.tau_gs, hard=hard, dim=-1)  # [M, L, N]

        # Compute truth values for each literal in each rule
        # mu_{m,j} = (1 - eta_{m,j}) * (gamma_{m,j} · c) + eta_{m,j} * (1 - (gamma_{m,j} · c))
        # This implements: mu = c if no negation, else mu = not c
        
        # gamma[c] selection: [M, L] @ [B, N].T -> [M, L, B]
        selected_conf = torch.einsum('mln,bn->mlb', gamma, c)  # [M, L, B]
        
        # Apply negation: mu = (1-eta) * selected + eta * (1 - selected)
        eta_expanded = self.eta.unsqueeze(-1).expand(-1, -1, B)  # [M, L, B]
        mu = (1 - eta_expanded) * selected_conf + eta_expanded * (1 - selected_conf)

        # Rule strengths: conjunction of all literals in rule
        # tau_m = AND_{j} mu_{m,j} -> product [M, B]
        tau = torch.prod(mu, dim=1)  # [M, B]

        # Class logits: sum over rules weighted by w
        # logits_y = beta + w @ tau [Y, B]
        logits_y = self.beta.unsqueeze(-1) + torch.matmul(self.w, tau)  # [Y, B]

        # Probabilities via softmax
        P = F.softmax(logits_y, dim=0)  # [Y, B]

        return P, gamma, tau

    def get_sparsity_loss(self):
        """Compute sparsity loss to encourage interpretable rules.
        
        Returns:
            Scalar tensor for regularization
        """
        # L_sparse = ||Gamma||_1 + ||w||_1
        gamma_l1 = torch.sum(torch.abs(self.Gamma))
        w_l1 = torch.sum(torch.abs(self.w))
        return gamma_l1 + w_l1