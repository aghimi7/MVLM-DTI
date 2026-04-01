import os, torch, numpy as np, pandas as pd, torch.nn as nn, torch.nn.functional as F
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
        # EXACT ORIGINAL ARCHITECTURE
        self.prot_proj = nn.utils.spectral_norm(nn.Linear(480, 512))
        
    def forward(self, p_emb):
        return F.normalize(self.prot_proj(p_emb), dim=-1)

def compute_knn_purity(k=5):
    print("--- COMPUTING MANIFOLD K-NN PURITY ---")
    prot_map = pd.read_csv(get_file("final_protein_map.csv"))
    prot_vectors = np.load(get_file("big_protein_vectors.npy"))

    families =[]
    for i, row in prot_map.iterrows():
        name = "".join([str(row[c]).upper() for c in['Entry Name', 'Gene Names', 'Target_ID'] if c in row and pd.notna(row[c])])
        if 'CDK' in name or 'MAPK' in name: families.append('CMGC')
        elif 'PKC' in name: families.append('AGC')
        elif any(x in name for x in['SRC', 'EGF', 'FGF', 'JAK']): families.append('TK')
        else: families.append('Other')
    
    prot_map['Family'] = families

    model = UniversalMVLM()
    state_dict = torch.load(get_file("MVLM_Foundation.pt"), map_location='cpu')
    
    # Safely extract only the protein projection weights
    clean_dict = {}
    for key, val in state_dict.items():
        k_clean = key.replace("module.", "")
        if 'prot_proj' in k_clean:
            clean_dict[k_clean] = val
            
    model.load_state_dict(clean_dict, strict=False)
    model.eval()

    with torch.no_grad():
        v_tensor = torch.tensor(prot_vectors, dtype=torch.float32)
        projected = model(v_tensor).numpy()

    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn_model.fit(projected)
    _, indices = nn_model.kneighbors(projected)
    neighbor_indices = indices[:, 1:]
    
    fam_arr = np.array(families)
    purities, tk_p, cmgc_p = [], [],[]

    for i, neighbors in enumerate(neighbor_indices):
        if fam_arr[i] == 'Other': continue
        p = np.sum(fam_arr[neighbors] == fam_arr[i]) / k
        purities.append(p)
        if fam_arr[i] == 'TK': tk_p.append(p)
        elif fam_arr[i] == 'CMGC': cmgc_p.append(p)

    print(f"\n[Phylogenetic Clustering Results (k={k})]")
    print(f"Global Purity: {np.mean(purities)*100:.1f}%")
    print(f"TK Family:     {np.mean(tk_p)*100:.1f}%")
    print(f"CMGC Family:   {np.mean(cmgc_p)*100:.1f}%")

if __name__ == "__main__":
    compute_knn_purity()
