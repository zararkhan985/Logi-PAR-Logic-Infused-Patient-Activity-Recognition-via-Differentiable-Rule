# Results from experiments on VAST and OmniFall benchmarks

vast_results = {
    'Logi-PAR': {'Acc': 93.5, 'mR@k': 91.2, 'm@P': 92.4, 'F1': 91.8, 'AUC': 0.96, 'F@R': 0.04},
    'Dual-Causal': {'Acc': 90.5, 'mR@k': 88.9, 'm@P': 90.2, 'F1': 89.5, 'AUC': 0.94, 'F@R': 0.06},
    'InternVideo2': {'Acc': 90.1, 'mR@k': 91.5, 'm@P': 78.2, 'F1': 84.3, 'AUC': 0.90, 'F@R': 0.18},
    # Add other methods as needed
}

omnifall_results = {
    'Logi-PAR': {'CGS': 89.4, 'NPR': 91.5, 'mAP': 93.1, 'CF@val': 88.2, 'mS@Acc': 90.5, 'F1': 91.0},
    'Dual-Causal': {'CGS': 82.5, 'NPR': 80.2, 'mAP': 89.3, 'CF@val': 78.4, 'mS@Acc': 86.2, 'F1': 88.4},
    'InternVideo2': {'CGS': 68.3, 'NPR': 62.5, 'mAP': 91.5, 'CF@val': 35.4, 'mS@Acc': 65.8, 'F1': 84.8},
    # Add other methods
}

# Ablation results (placeholder, based on text description)
ablation_results = {
    'Full Model': {'CGS': 89.4, 'F@R': 0.04, 'm@P': 92.4},
    'Variant A (No Fusion)': {'CGS': 65.0, 'F@R': 0.08, 'm@P': 86.5},
    'Variant B (No Logic)': {'CGS': 27.2, 'F@R': 0.18, 'm@P': 74.1},
    'Variant C (No Temporal)': {'CGS': 82.1, 'F@R': 0.14, 'm@P': 88.9},
}

if __name__ == '__main__':
    print("VAST Results:")
    for model, metrics in vast_results.items():
        print(f"{model}: {metrics}")
    print("\nOmniFall Results:")
    for model, metrics in omnifall_results.items():
        print(f"{model}: {metrics}")
    print("\nAblation Results:")
    for variant, metrics in ablation_results.items():
        print(f"{variant}: {metrics}")