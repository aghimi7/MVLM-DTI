"""
eval_mcnemar.py

Computes the McNemar's test statistic comparing the MVLM ensemble 
against a baseline proxy (e.g., InceptionDTA) to establish statistical 
significance of the decision boundaries.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

class MVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(768, 512, bias=False)
        self.V = nn.Linear(480, 512, bias=False)
        self.ln_d = nn.LayerNorm(512)
        self.ln_p = nn.LayerNorm(512)

    def forward(self, d, p):
        return torch.sum(F.normalize(self.ln_d(self.U(d)), dim=-1) * F.normalize(self.ln_p(self.V(p)), dim=-1), dim=1)

def optimize_threshold(probs, y_true):
    """Finds the optimal threshold maximizing F1-score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return (probs >= best_thresh).astype(int), best_thresh, best_f1

def run_mcnemar_test(data_dir="../data", models_dir="../models"):
    print("--- RUNNING MCNEMAR'S STATISTICAL TEST ---")
    
    # In a fully reproducible repo, predictions should ideally be generated live
    # or loaded from a standardized numpy archive.
    proxy_path = os.path.join(data_dir, "plot_data.npz")
    if not os.path.exists(proxy_path):
        print(f"Error: Baseline proxy predictions ({proxy_path}) not found.")
        return

    npz = np.load(proxy_path)
    y_proxy_prob = npz['y_proxy']
    y_ours_prob = npz['y_ours']
    y_true = npz['y_true']

    # Optimize thresholds for fair comparison
    my_preds, my_t, my_f1 = optimize_threshold(y_ours_prob, y_true)
    px_preds, px_t, px_f1 = optimize_threshold(y_proxy_prob, y_true)

    print(f"MVLM Ensemble Optimal F1: {my_f1:.4f} (Threshold: {my_t:.2f})")
    print(f"Baseline Proxy Optimal F1: {px_f1:.4f} (Threshold: {px_t:.2f})")

    # Construct contingency table
    both_correct = np.sum((my_preds == y_true) & (px_preds == y_true))
    only_ours_correct = np.sum((my_preds == y_true) & (px_preds != y_true))
    only_px_correct = np.sum((my_preds != y_true) & (px_preds == y_true))
    both_wrong = np.sum((my_preds != y_true) & (px_preds != y_true))

    table = [[both_correct, only_ours_correct],
             [only_px_correct, both_wrong]]

    print("\n[Contingency Table]")
    print(f"Both Correct:      {both_correct}")
    print(f"Only MVLM Correct: {only_ours_correct}")
    print(f"Only Proxy Correct:{only_px_correct}")
    print(f"Both Wrong:        {both_wrong}")

    result = mcnemar(table, exact=False, correction=True)
    
    print("\n[Statistical Results]")
    print(f"Chi-Square Statistic: {result.statistic:.4f}")
    print(f"P-Value:              {result.pvalue:.4e}")

if __name__ == "__main__":
    run_mcnemar_test()
