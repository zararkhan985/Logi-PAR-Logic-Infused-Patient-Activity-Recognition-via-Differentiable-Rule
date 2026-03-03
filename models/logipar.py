import torch
import torch.nn as nn
from .perception import MultiViewPerception
from .fusion import UncertaintyAwareFusion
from .reasoning import NeuralGuidedRuleLearner
from .explanation import CausalExplanation

class LogiPAR(nn.Module):
    def __init__(self, num_facts=24, num_views=4, num_rules=10, num_literals=3, num_classes=2):
        super().__init__()
        self.perception = MultiViewPerception(num_facts, num_views=num_views)
        self.fusion = UncertaintyAwareFusion(num_facts, num_views)
        self.reasoning = NeuralGuidedRuleLearner(num_facts, num_rules, num_literals, num_classes)
        self.explanation = CausalExplanation(self, num_facts)

    def forward(self, images, return_explanation=False):
        # images: list of [B, 3, 224, 224] for V views
        z, rho = self.perception(images)
        c, v = self.fusion(z, rho)
        P, gamma, tau = self.reasoning(c)

        if return_explanation:
            y_hat = torch.argmax(P, dim=0)
            delta, cf_score = self.explanation(c, y_hat)
            return P, c, v, gamma, tau, delta, cf_score
        else:
            return P, c, v, gamma, tau