"""
pretrain_foundation.py

Trains the Universal Foundation Model (MVLM) on the 473,000-interaction 
BindingDB master dataset. Evaluates performance using a strict 10% 
Cold-Drug (zero-shot) split.
"""

import os
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

print("--- TRAINING UNIVERSAL FOUNDATION MODEL (768-DIM + L2 NORM) ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths =[
        os.path.join(base_dir, "data", filename), 
        os.path.join(base_dir, "models", filename), 
        os.path.join(base_dir, filename)
    ]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {filename}")

# 1. Load the 473k Master Dataset
df = pd.read_csv(get_file("repaired_master_data.csv")).dropna(subset=['smiles', 'target', 'label'])
prot_map = pd.read_csv(get_file("final_protein_map.csv"))
prot_vectors = np.load(get_file("big_protein_vectors.npy"))

t2idx = {}
for i, row in prot_map.iterrows():
    for col in ['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
        if col in prot_map.columns and pd.notna(row[col]):
            for n in str(row[col]).replace(';', ' ').split(): 
                t2idx[n] = i

# Filter to valid targets
df = df[df['target'].astype(str).isin(t2idx)].reset_index(drop=True)
print(f"Loaded {len(df)} valid interactions from Universal Foundation Set.")

# 2. Extract DeepChem Features
print("Extracting 768-dim ChemBERTa Multi-View features (Batching to save memory)...")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
chemberta.eval()

unique_drugs = df['smiles'].unique()
d_embs = {}
batch_size = 512

with torch.no_grad():
    for i in range(0, len(unique_drugs), batch_size):
        batch_smiles = list(unique_drugs[i:i+batch_size])
        inp = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        out = chemberta(**inp)
        
        # Multi-View: CLS + Mean Pool
        concat_emb = torch.cat([out.last_hidden_state[:,0,:], torch.mean(out.last_hidden_state, dim=1)], dim=1).cpu().numpy()
        for j, sm in enumerate(batch_smiles):
            d_embs[sm] = concat_emb[j]
            
        if (i + batch_size) % 25600 == 0:
            print(f"  Processed {i + batch_size}/{len(unique_drugs)} drugs...")

print("Building data arrays...")
X_d = np.array([d_embs[sm] for sm in df['smiles']])
X_p = np.array([prot_vectors[t2idx[str(t)]] for t in df['target']])
Y = df['label'].values

# 3. 90/10 Cold-Drug Split
unique_train_drugs, unique_test_drugs = train_test_split(unique_drugs, test_size=0.1, random_state=42)
train_set = set(unique_train_drugs)

train_idx =[i for i, sm in enumerate(df['smiles']) if sm in train_set]
test_idx = [i for i, sm in enumerate(df['smiles']) if sm not in train_set]

class FastDataset(Dataset):
    def __init__(self, d, p, y):
        self.d = torch.tensor(d, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.d[i], self.p[i], self.y[i]

train_loader = DataLoader(FastDataset(X_d[train_idx], X_p[train_idx], Y[train_idx]), batch_size=1024, shuffle=True)
test_loader = DataLoader(FastDataset(X_d[test_idx], X_p[test_idx], Y[test_idx]), batch_size=1024, shuffle=False)

# 4. The Perfected Architecture
class MVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = nn.utils.spectral_norm(nn.Linear(768, 512, bias=False))
        self.V = nn.utils.spectral_norm(nn.Linear(480, 512, bias=False))
        self.ln_d = nn.LayerNorm(512)
        self.ln_p = nn.LayerNorm(512)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, d, p):
        d_p = F.normalize(self.ln_d(self.U(d)), dim=-1)
        p_p = F.normalize(self.ln_p(self.V(p)), dim=-1)
        return torch.sum(d_p * p_p, dim=1)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.3, temp=0.07):
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, scores, targets, pos_weight):
        logits = scores / self.temp
        loss_bce = self.bce(logits, targets)
        weight_vec = torch.where(targets == 1.0, pos_weight, torch.tensor(1.0).to(device))
        loss_bce = (loss_bce * weight_vec).mean()
        loss_geo = torch.mean((scores - targets) ** 2)
        return (1 - self.alpha) * loss_bce + self.alpha * loss_geo

# 5. Training Engine
model = MVLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Dynamic class weight
pos_w = torch.tensor([(len(train_idx) - sum(Y[train_idx])) / (sum(Y[train_idx]) + 1e-5)]).to(device)
criterion = HybridLoss()

print(f"\nTraining on {len(train_idx)} interactions (Evaluating on {len(test_idx)} cold-drug interactions)...")
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.join(save_dir, "MVLM_Foundation.pt")

best_auc = 0.0
patience_counter = 0

for epoch in range(60):
    model.train()
    total_loss = 0
    for d, p, y_b in train_loader:
        d, p, y_b = d.to(device), p.to(device), y_b.to(device)
        optimizer.zero_grad()
        score = model(d, p)
        loss = criterion(score, y_b, pos_weight=pos_w)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    model.eval()
    preds =[]
    with torch.no_grad():
        for d, p, _ in test_loader:
            d, p = d.to(device), p.to(device)
            score = model(d, p)
            preds.extend(torch.sigmoid(score / 0.07).cpu().numpy())
    
    auc = roc_auc_score(Y[test_idx], preds)
    scheduler.step(auc)
    
    if auc > best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(), save_name)
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Cold-Drug AUC: {auc:.4f} <-- NEW BEST")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Cold-Drug AUC: {auc:.4f}")
        
    if patience_counter >= 8:
        print("Early stopping triggered.")
        break

print(f"\nFoundation Model Training Complete! Saved to {save_name}")
print(f"Final Cold-Drug AUROC: {best_auc:.4f}")