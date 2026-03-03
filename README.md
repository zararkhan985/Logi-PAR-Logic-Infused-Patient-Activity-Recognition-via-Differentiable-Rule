# Logi-PAR: Logic-Infused Framework for Clinical Risk Assessment

A neuro-symbolic framework that factorizes clinical risk assessment into perception and reasoning stages, enabling interpretable and robust multi-view activity recognition in healthcare settings.

## Overview

Logi-PAR combines deep learning perception with differentiable logical reasoning to provide:

- **Perception Module**: Multi-view fact extraction using Swin-Transformer with uncertainty-aware fusion
- **Reasoning Module**: Differentiable logic rules with Gumbel-Softmax for rule learning and negation
- **Causal Explanations**: Counterfactual analysis for minimal perturbations

## Installation

1. Clone the repository:
```bash


2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support, ensure you have CUDA installed. The code automatically uses GPU if available.

## Project Structure

```
LOGI_PAR/
├── main.py                 # Entry point for training
├── train.py               # Training script with curriculum learning
├── results.py             # Results visualization
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── models/
│   ├── __init__.py
│   ├── logipar.py         # Full LogiPAR model
│   ├── perception.py      # Multi-view perception module
│   ├── fusion.py          # Uncertainty-aware fusion
│   ├── reasoning.py        # Neural-guided rule learner
│   └── explanation.py     # Causal explanation module
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset loaders
├── utils/
│   ├── __init__.py
│   ├── losses.py          # Loss functions
│   └── metrics.py         # Evaluation metrics
└── results.json           # Experimental results
```

## Usage

### Training

Run the full training pipeline:
```bash
python main.py
```

Or run directly with train.py:
```bash
python train.py
```

The training script supports custom hyperparameters:
- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 1e-4)
- `warmup_epochs`: Epochs with frozen reasoning (default: 20)

### Evaluation

View experimental results:
```bash
python results.py
```

### Custom Training

```python
from train import train_logipar

# Custom training with different parameters
train_logipar(num_epochs=50, batch_size=16, lr=5e-5, warmup_epochs=10)
```

## Model Architecture

### Perception Module
- Uses Swin-Large as backbone (pretrained on ImageNet)
- Separate prediction and reliability heads for each clinical fact
- Supports arbitrary number of views (default: 4)

### Fusion Module
- Uncertainty-aware view weighting
- Reliability scores determine view contributions
- Outputs fused fact probabilities

### Reasoning Module
- Learnable logical rules with Gumbel-Softmax
- Supports literal negation
- Sparse rule weights for interpretability

### Explanation Module
- Counterfactual generation via gradient-based optimization
- Finds minimal perturbations that change predictions
- Provides interpretable explanations for clinical decisions

## Datasets

The framework supports two clinical datasets:

- **OmniFall**: Controlled multi-view fall detection dataset
- **VAST**: Real clinical data from hospital surveillance

Replace the dummy datasets in `data/dataset.py` with actual data loaders for your experiments.

## Results

Logi-PAR achieves state-of-the-art performance on clinical benchmarks:

| Dataset | Metric | Logi-PAR | Best Baseline |
|---------|--------|----------|---------------|
| VAST | Accuracy | 93.5% | 90.5% |
| VAST | F1 Score | 91.8% | 89.5% |
| VAST | AUC | 0.96 | 0.94 |
| OmniFall | CGS | 89.4% | 82.5% |
| OmniFall | F1 | 91.0% | 88.4% |

See `results.json` for complete results.

## Key Features

- Multi-view fusion with reliability weighting
- Differentiable rule learning with negation
- Curriculum training for stable convergence
- Causal counterfactual explanations
- State-of-the-art performance on clinical benchmarks
- Fully interpretable logical rules

