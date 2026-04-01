import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

def get_file(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths =[
        os.path.join(base_dir, "data", filename),
        os.path.join(base_dir, "models", filename),
        os.path.join(base_dir, filename)
    ]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {filename}")

class UniversalMVLM(nn.Module):
    def __init__(self):
        super().__init__()
        # EXACT architecture of the Foundation Model (with biases, NO LayerNorm)
        self.drug_proj = nn.utils.spectral_norm(nn.Linear(768, 512))
        self.prot_proj = nn.utils.spectral_norm(nn.Linear(480, 512))

    def forward(self, d_emb, p_emb):
        d_p = F.normalize(self.drug_proj(d_emb), dim=-1)
        p_p = F.normalize(self.prot_proj(p_emb), dim=-1)
        return torch.sum(d_p * p_p, dim=1)

class KinomeDiscoveryEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(self.device)
        self.chemberta.eval()

        self.prot_map = pd.read_csv(get_file("final_protein_map.csv"))
        prot_vectors = np.load(get_file("big_protein_vectors.npy"))
        self.p_tensor = torch.tensor(prot_vectors, dtype=torch.float32).to(self.device)

        self.model = UniversalMVLM().to(self.device)
        state_dict = torch.load(get_file("MVLM_Foundation.pt"), map_location=self.device)
        
        # Clean dictionary keys
        clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()

        with torch.no_grad():
            self.p_proj = F.normalize(self.model.prot_proj(self.p_tensor), dim=-1)

    def predict(self, smiles, top_k=5):
        with torch.no_grad():
            inp = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            out = self.chemberta(**inp)
            d_emb = torch.cat([out.last_hidden_state[:, 0, :], torch.mean(out.last_hidden_state, dim=1)], dim=1)
            
            d_proj = F.normalize(self.model.drug_proj(d_emb), dim=-1)
            scores = torch.matmul(d_proj, self.p_proj.T).squeeze(0)
            probs = torch.sigmoid(scores / 0.07).cpu().numpy()

        ranked_indices = np.argsort(probs)[::-1]
        
        results = []
        for idx in ranked_indices[:top_k]:
            target_id = self.prot_map.iloc[idx].get('Target_ID', self.prot_map.iloc[idx].get('Entry Name', 'Unknown'))
            results.append((target_id, probs[idx]))
        return results

if __name__ == "__main__":
    engine = KinomeDiscoveryEngine()
    test_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    
    print("\n[Discovery Engine Validation]")
    print(f"Query SMILES: {test_smiles[:30]}...")
    predictions = engine.predict(test_smiles, top_k=5)
    
    print(f"{'Target ID':<15} | {'Confidence':<10}")
    print("-" * 30)
    for target, prob in predictions:
        print(f"{target:<15} | {prob:.4f}")
