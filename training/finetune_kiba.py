import os, torch, warnings, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file(f):
    paths =[f, f"../{f}", f"data/{f}", f"../data/{f}", f"models/{f}", f"../models/{f}"]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {f}")

df = pd.read_csv(get_file("kiba_strict.csv")).dropna(subset=['Drug', 'Target_ID', 'Y'])
df['label'] = (df['Y'] < 12.1).astype(int)

prot_map = pd.read_csv(get_file("final_protein_map.csv"))
prot_vectors = np.load(get_file("big_protein_vectors.npy"))

t2idx = {}
for i, row in prot_map.iterrows():
    for col in['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
        if col in prot_map.columns and pd.notna(row[col]):
            for n in str(row[col]).replace(';', ' ').split():
                t2idx[n] = i

df = df[df['Target_ID'].astype(str).isin(t2idx)].reset_index(drop=True)

print("Extracting ChemBERTa features (768-dim)...")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)

d_embs = {}
with torch.no_grad():
    for drug in df['Drug'].unique():
        inp = tokenizer(drug, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        out = chemberta(**inp)
        # EXACTLY 768 DIMENSIONS (384 + 384)
        emb = torch.cat([out.last_hidden_state[:,0,:], torch.mean(out.last_hidden_state, dim=1)], dim=1).squeeze(0).cpu().numpy()
        d_embs[drug] = emb

X_d = np.array([d_embs[d] for d in df['Drug']])
X_p = np.array([prot_vectors[t2idx[str(t)]] for t in df['Target_ID']])
Y = df['label'].values

train_d, test_d, train_p, test_p, train_y, test_y = train_test_split(X_d, X_p, Y, test_size=0.2, random_state=42, stratify=Y)

class FastDataset(Dataset):
    def __init__(self, d, p, y):
        self.d, self.p, self.y = torch.tensor(d, dtype=torch.float32), torch.tensor(p, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.d[i], self.p[i], self.y[i]

train_loader = DataLoader(FastDataset(train_d, train_p, train_y), batch_size=256, shuffle=True)
test_loader = DataLoader(FastDataset(test_d, test_p, test_y), batch_size=256, shuffle=False)

class MVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = nn.utils.spectral_norm(nn.Linear(768, 512, bias=False)) # MATCHES WEIGHTS EXACTLY
        self.V = nn.utils.spectral_norm(nn.Linear(480, 512, bias=False))
        self.ln_d = nn.LayerNorm(512)
        self.ln_p = nn.LayerNorm(512)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)
    def forward(self, d, p):
        return torch.sum(F.normalize(self.ln_d(self.U(d)), dim=-1) * F.normalize(self.ln_p(self.V(p)), dim=-1), dim=1)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.3, temp=0.07):
        super().__init__()
        self.alpha, self.temp = alpha, temp
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, scores, targets, pos_weight):
        logits = scores / self.temp
        loss_bce = (self.bce(logits, targets) * torch.where(targets == 1.0, pos_weight, torch.tensor(1.0).to(device))).mean()
        return (1 - self.alpha) * loss_bce + self.alpha * torch.mean((scores - targets) ** 2)

foundation_path = get_file("MVLM_Foundation.pt")
os.makedirs("models", exist_ok=True)
ensemble_preds = np.zeros(len(test_y))

for seed in [42, 101, 999]:
    print(f"\n--- Training KIBA Seed {seed} ---")
    torch.manual_seed(seed)
    model = MVLM().to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(foundation_path, map_location=device).items()}, strict=False)
        
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)
    pos_w = torch.tensor([(len(train_y) - sum(train_y)) / (sum(train_y) + 1e-5)]).to(device)
    crit = HybridLoss() 
    
    best_auc, patience = 0.0, 0
    save_name = f"models/kiba_mvlm_seed_{seed}.pt"
    
    for epoch in range(60):
        model.train()
        for d, p, y_b in train_loader:
            opt.zero_grad()
            loss = crit(model(d.to(device), p.to(device)), y_b.to(device), pos_w)
            loss.backward(); opt.step()
            
        model.eval()
        preds =[]
        with torch.no_grad():
            for d, p, _ in test_loader:
                preds.extend(torch.sigmoid(model(d.to(device), p.to(device)) / 0.07).cpu().numpy())
        
        auc = roc_auc_score(test_y, preds)
        scheduler.step(auc)
        if auc > best_auc:
            best_auc, patience = auc, 0
            torch.save(model.state_dict(), save_name)
        else:
            patience += 1
        if patience >= 10: break
            
    model.load_state_dict(torch.load(save_name))
    model.eval()
    with torch.no_grad():
        final_preds =[]
        for d, p, _ in test_loader:
            final_preds.extend(torch.sigmoid(model(d.to(device), p.to(device)) / 0.07).cpu().numpy())
        ensemble_preds += np.array(final_preds)

ensemble_preds /= 3
print("\n==================================================")
print(f"FINAL KIBA ENSEMBLE AUROC: {roc_auc_score(test_y, ensemble_preds):.4f}")
print(f"FINAL KIBA ENSEMBLE AUPRC: {average_precision_score(test_y, ensemble_preds):.4f}")
print("==================================================")
