import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.logipar import LogiPAR
from utils.losses import LogiPARLoss
from data.dataset import DummyClinicalDataset
from tqdm import tqdm

def train_logipar(num_epochs=100, batch_size=32, lr=1e-4, warmup_epochs=20):
    """Train the LogiPAR model with curriculum learning.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        warmup_epochs: Number of warmup epochs where reasoning is frozen
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = LogiPAR().to(device)
    loss_fn = LogiPARLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Dataset
    dataset = DummyClinicalDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Curriculum: Freeze reasoning for warmup
    for param in model.reasoning.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        if epoch == warmup_epochs:
            # Unfreeze reasoning
            for param in model.reasoning.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch + 1}: Unfreezing reasoning module")

        model.train()
        total_loss = 0
        num_batches = 0
        
        for imgs_list, facts, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs_list = [img.to(device) for img in imgs_list]
            facts = facts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass through the full model
            P, c, v, gamma, tau = model(imgs_list)
            
            # Compute loss
            loss, l_ce, l_fact, l_sparse = loss_fn(P, labels, c, facts, model.reasoning)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model
    torch.save(model.state_dict(), 'logipar.pth')
    print("Model saved to logipar.pth")

if __name__ == '__main__':
    train_logipar()