import torch
import torch.nn as nn
import timm

class MultiViewPerception(nn.Module):
    def __init__(self, num_facts=24, embed_dim=1024, num_views=4):
        super().__init__()
        # Backbone: Swin-L pre-trained
        self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=0)
        self.embed_dim = embed_dim  # Assuming backbone outputs 1024-dim

        # Prediction heads: for each fact, pred and rel
        self.pred_heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_facts)])
        self.rel_heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_facts)])

        self.num_facts = num_facts
        self.num_views = num_views

    def forward(self, images):
        # images: list of tensors [V, 3, 224, 224]
        V = len(images)
        batch_size = images[0].shape[0] if images[0].dim() > 3 else 1  # Assume batch first

        # For each view, extract features
        features = []
        for img in images:
            if img.dim() == 3:  # Single image
                img = img.unsqueeze(0)
            feat = self.backbone(img)  # [B, embed_dim]
            features.append(feat)

        # Stack: [V, B, embed_dim]
        features = torch.stack(features, dim=0)

        # For each fact k
        z = torch.zeros(V, batch_size, self.num_facts)  # z_k^{(v)}
        rho = torch.zeros(V, batch_size, self.num_facts)  # rho_k^{(v)}

        for k in range(self.num_facts):
            for v in range(V):
                z[v, :, k] = self.pred_heads[k](features[v]).squeeze(-1)
                rho[v, :, k] = self.rel_heads[k](features[v]).squeeze(-1)

        return z, rho  # [V, B, N], [V, B, N]