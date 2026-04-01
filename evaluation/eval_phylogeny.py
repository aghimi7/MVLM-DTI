"""
eval_phylogeny.py

Evaluates the interpretability of the latent protein manifold by computing 
k-Nearest Neighbor (k-NN) purity for structural kinase families.
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class UniversalMVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prot_proj = nn.utils.spectral_norm(nn.Linear(480, 512))

    def forward(self, p_emb):
        return F.normalize(self.prot_proj(p_emb), dim=-1)

def compute_knn_purity(data_dir="../data", model_path="../models/MVLM_Foundation.pt", k=5):
    print("--- COMPUTING MANIFOLD K-NN PURITY ---")
    
    prot_map = pd.read_csv(os.path.join(data_dir, "final_protein_map.csv"))
    prot_vectors = np.load(os.path.join(data_dir, "big_protein_vectors.npy"))

    # Heuristic family mapping for known evolutionary clusters
    families =[]
    for name in prot_map['Target_ID']:
        n = str(name).upper()
        if 'CDK' in n: families.append('CMGC')
        elif 'MAPK' in n: families.append('CMGC')
        elif 'PKC' in n: families.append('AGC')
        elif 'SRC' in n: families.append('TK')
        elif 'EGF' in n: families.append('TK')
        elif 'FGF' in n: families.append('TK')
        elif 'JAK' in n: families.append('TK')
        else: families.append('Other')
    
    prot_map['Family'] = families

    # Load Model
    model = UniversalMVLM()
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if 'prot_proj' in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        v_tensor = torch.tensor(prot_vectors, dtype=torch.float32)
        projected = model(v_tensor).numpy()

    # Calculate k-NN Purity
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn_model.fit(projected)
    _, indices = nn_model.kneighbors(projected)

    # Exclude the point itself (index 0)
    neighbor_indices = indices[:, 1:]
    
    family_labels = np.array(families)
    purities =[]
    tk_purities, cmgc_purities = [],[]

    for i, neighbors in enumerate(neighbor_indices):
        true_family = family_labels[i]
        if true_family == 'Other':
            continue
            
        neighbor_families = family_labels[neighbors]
        matches = np.sum(neighbor_families == true_family)
        purity = matches / k
        
        purities.append(purity)
        if true_family == 'TK': tk_purities.append(purity)
        elif true_family == 'CMGC': cmgc_purities.append(purity)

    print(f"\n[Phylogenetic Clustering Results (k={k})]")
    print(f"Global Purity: {np.mean(purities)*100:.1f}%")
    print(f"TK Family:     {np.mean(tk_purities)*100:.1f}%")
    print(f"CMGC Family:   {np.mean(cmgc_purities)*100:.1f}%")

if __name__ == "__main__":
    compute_knn_purity()
