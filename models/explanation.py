import torch
import torch.nn as nn

class CausalExplanation(nn.Module):
    """Causal explanation module for counterfactual analysis.
    
    Generates minimal perturbations to change model predictions,
    providing interpretable explanations for clinical risk assessments.
    """
    
    def __init__(self, model, num_facts=24, max_iter=100, lr=0.1):
        super().__init__()
        self.model = model  # The full LogiPAR model
        self.num_facts = num_facts
        self.max_iter = max_iter
        self.lr = lr

    def forward(self, c, y_hat):
        """Generate counterfactual explanations.
        
        Args:
            c: [B, N] - Grounded facts from perception
            y_hat: [B] - Predicted class indices
            
        Returns:
            delta: [B, N] - Minimal perturbations
            l1_norm: [B] - L1 norm of perturbations
        """
        B = c.shape[0]
        
        # Clone c to avoid modifying the original
        c_original = c.detach()
        delta = torch.zeros_like(c_original, requires_grad=True)  # [B, N]

        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            c_pert = c_original + delta
            
            # Get predictions from reasoning module directly
            # Pass c_pert as the fused facts to reasoning
            P, _, _ = self.model.reasoning(c_pert)
            y_pert = torch.argmax(P, dim=0)  # [B]
            
            # Minimize number of samples that still have same prediction
            loss = torch.sum((y_pert == y_hat).float())
            
            # Add L1 regularization to encourage minimal changes
            loss = loss + 0.01 * torch.sum(torch.abs(delta))
            
            loss.backward()
            optimizer.step()
            
            # Clamp delta to [-1, 1] and round to integers {-1, 0, 1}
            with torch.no_grad():
                delta.clamp_(-1, 1)
                delta.round_()

        # Compute L1 norm
        l1_norm = torch.sum(torch.abs(delta), dim=1)  # [B]

        return delta, l1_norm