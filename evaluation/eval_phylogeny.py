import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def get_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths =[os.path.join(base_dir, "data", filename), os.path.join(base_dir, "models", filename)]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {filename}")

class UniversalMVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prot_proj = nn.utils.spectral_norm(nn.Linear(480, 512, bias=False))
        self.ln_p = nn.LayerNorm(512)

    def forward(self, p_emb):
        return F.normalize(self.ln_p(self.prot_proj(p_emb)), dim=-1)

def compute_knn_purity(k=5):
    print("--- COMPUTING MANIFOLD K-NN PURITY ---")
    prot_map = pd.read_csv(get_file("final_protein_map.csv"))
    prot_vectors = np.load(get_file("big_protein_vectors.npy"))

    families =[]
    for i, row in prot_map.iterrows():
        name = ""
        for col in ['Entry Name', 'Gene Names', 'Target_ID', 'Name']:
            if col in row and pd.notna(row[col]):
                name += str(row[col]).upper()
        
        if 'CDK' in name or 'MAPK' in name: families.append('CMGC')
        elif 'PKC' in name: families.append('AGC')
        elif 'SRC' in name or 'EGF' in name or 'FGF' in name or 'JAK' in name: families.append('TK')
        else: families.append('Other')
    
    prot_map['Family'] = families

    model = UniversalMVLM()
    state_dict = torch.load(get_file("MVLM_Foundation.pt"), map_location='cpu')
    state_dict = {k.replace("module.", "").replace("V", "prot_proj"): v for k, v in state_dict.items() if 'V' in k or 'ln_p' in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        v_tensor = torch.tensor(prot_vectors, dtype=torch.float32)
        projected = model(v_tensor).numpy()

    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn_model.fit(projected)
    _, indices = nn_model.kneighbors(projected)
    neighbor_indices = indices[:, 1:]
    
    family_labels = np.array(families)
    purities, tk_purities, cmgc_purities = [], [], []

    for i, neighbors in enumerate(neighbor_indices):
        true_family = family_labels[i]
        if true_family == 'Other': continue
            
        matches = np.sum(family_labels[neighbors] == true_family)
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
