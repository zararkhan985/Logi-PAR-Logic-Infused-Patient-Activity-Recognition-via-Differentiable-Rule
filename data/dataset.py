import torch
from torch.utils.data import Dataset

class DummyClinicalDataset(Dataset):
    def __init__(self, num_samples=1000, num_views=4, num_facts=24, num_classes=2):
        self.num_samples = num_samples
        self.num_views = num_views
        self.num_facts = num_facts
        self.num_classes = num_classes

        # Generate dummy data
        self.images = [torch.randn(num_samples, 3, 224, 224) for _ in range(num_views)]
        self.fact_labels = torch.randint(0, 2, (num_samples, num_facts)).float()  # Binary facts
        self.class_labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imgs = [img[idx] for img in self.images]
        facts = self.fact_labels[idx]
        label = self.class_labels[idx]
        return imgs, facts, label

# For OmniFall and VAST, inherit or modify
class OmniFallDataset(DummyClinicalDataset):
    pass

class VASTDataset(DummyClinicalDataset):
    pass