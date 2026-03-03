import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_metrics(y_pred, y_true, probs=None):
    # y_pred, y_true: numpy arrays
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, probs[:, 1]) if probs is not None and len(torch.unique(y_true)) > 1 else None
    return {'acc': acc, 'f1': f1, 'auc': auc}

# Compositional Generalization Score (CGS): placeholder, assume some calculation
def compute_cgs(predictions, labels, compositional_splits):
    # Dummy: accuracy on compositional splits
    return accuracy_score(labels, predictions)

# Other metrics as needed