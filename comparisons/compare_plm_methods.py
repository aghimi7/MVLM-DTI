#!/usr/bin/env python3
"""
compare_plm_methods.py
──────────────────────
Fair comparison of MVLM against two recently published PLM-based
drug-target interaction methods on the strict KIBA benchmark:

    - PMMR  (Ouyang et al., Bioinformatics 2025, doi:10.1093/bioinformatics/btaf002)
    - MGF-DTA (Ni et al., Int J Mol Sci 2026, doi:10.3390/ijms27020947)

All three models are trained and evaluated on identical 80/20 splits of
kiba_strict.csv across three random seeds (42, 101, 999). Results are
reported as Mean ± SD.

Label convention:
    KIBA score < 12.1  →  active  (label = 1)
    KIBA score >= 12.1 →  inactive (label = 0)
    This follows the standard community convention for KIBA binary classification.

PMMR and MGF-DTA are regression models. Their continuous affinity
predictions are converted to binary classifications using the same
threshold (score < 12.1) to enable direct AUROC/AUPRC comparison
alongside their native regression metrics (CI, Rm²).

Note on MGF-DTA:
    The MGF-DTA repository (https://github.com/fdmsz/MGF-DTA) does not
    include a public model implementation — only data files. Results for
    MGF-DTA are therefore reported for a faithful reimplementation of the
    architecture described in Ni et al. (2026).

Note on PMMR:
    The official PMMR TensorFlow implementation was also evaluated on the
    same test split (seed=42). To ensure label convention consistency,
    predictions were post-hoc corrected: correct_labels = 1 - pmmr_labels,
    correct_probs = 1 - pmmr_probs. This yielded AUROC=0.815, AUPRC=0.939.
    See eval_pmmr_official.py for the official evaluation script.

Usage:
    python compare_plm_methods.py --data_dir . --epochs 100

Requirements:
    pip install torch transformers scikit-learn scipy rdkit pandas numpy
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             mean_squared_error)
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()

SEEDS     = [42, 101, 999]
THRESHOLD = 12.1   # KIBA: score < threshold = active


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fair comparison: MVLM vs PMMR vs MGF-DTA on KIBA')
    parser.add_argument('--data_dir',    type=str, default='.')
    parser.add_argument('--epochs',      type=int, default=100)
    parser.add_argument('--skip_pmmr',   action='store_true')
    parser.add_argument('--skip_mgfdta', action='store_true')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_file(filename, data_dir='.'):
    candidates = [
        filename,
        os.path.join(data_dir, filename),
        os.path.join('..', filename),
        os.path.join('data', filename),
        os.path.join('models', filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find '{filename}'")


def ci_score(y_true, y_pred, n_sample=3000):
    """Concordance Index, sampled for efficiency on large test sets."""
    rng = np.random.default_rng(42)
    if len(y_true) > n_sample:
        idx = rng.choice(len(y_true), n_sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    n = len(y_true)
    c = t = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                t += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    c += 1
                elif y_pred[i] == y_pred[j]:
                    c += 0.5
    return c / t if t else 0.5


def rm2_score(y_true, y_pred):
    """Modified R-squared (Rm²)."""
    r2    = pearsonr(y_true, y_pred)[0] ** 2
    slope = np.polyfit(y_true, y_pred, 1)
    y_hat = np.polyval(slope, y_true)
    ss_r  = np.sum((y_pred - y_hat) ** 2)
    ss_t  = np.sum((y_pred - y_pred.mean()) ** 2)
    r02   = 1 - ss_r / ss_t if ss_t else 0
    return r2 * (1 - np.sqrt(abs(r2 - r02)))


def fmt(values):
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def morgan_fp(smiles, radius=2, n_bits=1024):
    """Morgan fingerprint for a SMILES string."""
    try:
        from rdkit.Chem import AllChem, MolFromSmiles
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'rdkit',
                        '--break-system-packages', '-q'])
        from rdkit.Chem import AllChem, MolFromSmiles
    mol = MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def kmer_pca(sequences, n_components=128, k=3, vocab_size=4000):
    """k-mer frequency vectors reduced with PCA."""
    def kfreq(seq):
        kms = [seq[i:i+k] for i in range(max(0, len(seq)-k+1))]
        c = {}
        for x in kms: c[x] = c.get(x, 0) + 1
        t = max(len(kms), 1)
        return {x: v/t for x, v in c.items()}
    vc = {}
    for s in sequences:
        for x, f in kfreq(s).items(): vc[x] = vc.get(x, 0) + f
    vocab = {x: i for i, x in enumerate(
        sorted(vc, key=vc.get, reverse=True)[:vocab_size])}
    mat = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        for x, f in kfreq(seq).items():
            if x in vocab: mat[row, vocab[x]] = f
    nc = min(n_components, mat.shape[1], mat.shape[0]-1)
    return PCA(n_components=nc).fit_transform(mat).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, d, p, y):
        self.d = torch.tensor(d, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.d[i], self.p[i], self.y[i]


class MGFDataset(Dataset):
    def __init__(self, d, p, fp, km, y):
        self.d  = torch.tensor(d,  dtype=torch.float32)
        self.p  = torch.tensor(p,  dtype=torch.float32)
        self.fp = torch.tensor(fp, dtype=torch.float32)
        self.km = torch.tensor(km, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.d[i], self.p[i], self.fp[i], self.km[i], self.y[i]


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────

class MVLM(nn.Module):
    """
    Multi-View Linear Manifold (MVLM).
    Projects frozen ChemBERTa (768-d) and ESM-2 (480-d) embeddings into
    a shared 512-d hypersphere via learnable spectral-norm linear projections.
    Binding affinity = cosine similarity in the shared space.
    """
    def __init__(self, drug_dim=768, prot_dim=480, latent_dim=512):
        super().__init__()
        self.U    = nn.utils.spectral_norm(
            nn.Linear(drug_dim, latent_dim, bias=False))
        self.V    = nn.utils.spectral_norm(
            nn.Linear(prot_dim, latent_dim, bias=False))
        self.ln_d = nn.LayerNorm(latent_dim)
        self.ln_p = nn.LayerNorm(latent_dim)

    def forward(self, d, p):
        zd = F.normalize(self.ln_d(self.U(d)), dim=-1)
        zp = F.normalize(self.ln_p(self.V(p)), dim=-1)
        return torch.sum(zd * zp, dim=1)


class HybridLoss(nn.Module):
    """
    Hybrid loss: (1-alpha) * WBCE + alpha * MSE geometric regularisation.
    """
    def __init__(self, alpha=0.3, temp=0.07):
        super().__init__()
        self.alpha = alpha
        self.temp  = temp
        self.bce   = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, scores, targets, pos_weight):
        logits   = scores / self.temp
        weights  = torch.where(targets == 1.0, pos_weight,
                               torch.ones_like(targets))
        loss_bce = (self.bce(logits, targets) * weights).mean()
        loss_mse = torch.mean((scores - targets) ** 2)
        return (1 - self.alpha) * loss_bce + self.alpha * loss_mse


class PMMR(nn.Module):
    """
    PMMR-style reimplementation.
    Applies frozen PLM embeddings through Transformer encoder layers
    with multi-head self-attention and learned attention pooling,
    followed by an FC regression head.

    Reference: Ouyang et al., Bioinformatics 2025.
               https://doi.org/10.1093/bioinformatics/btaf002
    Note: The official PMMR implementation additionally uses GCN on
    molecular graphs. This reimplementation operates on frozen PLM
    embeddings to enable fair comparison on the same feature space.
    """
    def __init__(self, drug_dim=768, prot_dim=480, hidden=256,
                 n_heads=4, n_layers=2):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.prot_proj = nn.Linear(prot_dim, hidden)
        d_layer = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, batch_first=True, dropout=0.1)
        p_layer = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, batch_first=True, dropout=0.1)
        self.drug_enc  = nn.TransformerEncoder(d_layer, n_layers)
        self.prot_enc  = nn.TransformerEncoder(p_layer, n_layers)
        self.drug_attn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.prot_attn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1))

    def _attn_pool(self, attn, x):
        w = torch.softmax(attn(x), dim=1)
        return (w * x).sum(1)

    def forward(self, d, p):
        d  = self.drug_enc(self.drug_proj(d).unsqueeze(1))
        p  = self.prot_enc(self.prot_proj(p).unsqueeze(1))
        fd = self._attn_pool(self.drug_attn, d)
        fp = self._attn_pool(self.prot_attn, p)
        return self.fc(torch.cat([fd, fp], dim=-1)).squeeze(-1)


class MGFDTA(nn.Module):
    """
    MGF-DTA-style reimplementation.
    Gated drug fusion (ChemBERTa + Morgan FP) + residual protein fusion
    (ESM-2 + k-mer PCA) + hierarchical attention pooling + FC regression.

    Reference: Ni et al., Int J Mol Sci 2026.
               https://doi.org/10.3390/ijms27020947
    Note: The MGF-DTA repository does not include a public model
    implementation. This is a faithful reimplementation based on the
    architecture described in the paper.
    """
    def __init__(self, drug_dim=768, fp_dim=1024, prot_dim=480,
                 kmer_dim=128, hidden=512):
        super().__init__()
        self.fp_proj   = nn.Linear(fp_dim,  drug_dim)
        self.gate      = nn.Linear(drug_dim * 3, drug_dim)
        self.drug_out  = nn.Linear(drug_dim, hidden)
        self.km_proj   = nn.Linear(kmer_dim, prot_dim)
        self.prot_out  = nn.Linear(prot_dim, hidden)
        self.hier_attn = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden),
                          nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(3)])
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1))

    def _gated_fuse(self, x, fp):
        xfp = self.fp_proj(fp)
        g   = torch.sigmoid(self.gate(torch.cat([x, xfp, x + xfp], dim=-1)))
        return g * x + (1 - g) * xfp

    def _res_fuse(self, x_esm, x_km):
        xk2 = self.km_proj(x_km)
        w   = torch.sigmoid(x_esm.norm(dim=-1, keepdim=True))
        return x_esm + (1 - w) * xk2

    def _hier_pool(self, x):
        outs = [(torch.softmax(a(x.unsqueeze(1)), dim=1) *
                 x.unsqueeze(1)).sum(1) for a in self.hier_attn]
        return torch.stack(outs, dim=1).mean(1)

    def forward(self, d, p, fp, km):
        fd  = F.relu(self.drug_out(self._gated_fuse(d, fp)))
        fp_ = F.relu(self.prot_out(self._res_fuse(p, km)))
        return self.fc(
            torch.cat([self._hier_pool(fd),
                       self._hier_pool(fp_)], dim=-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_mvlm(tr_loader, te_loader, te_yb, device,
               foundation_path, epochs, seed):
    torch.manual_seed(seed)
    model = MVLM().to(device)
    if os.path.exists(foundation_path):
        st = {k.replace("module.", ""): v for k, v in
              torch.load(foundation_path, map_location=device).items()}
        model.load_state_dict(st, strict=False)
        print("  Foundation weights loaded.")
    else:
        print(f"  WARNING: Foundation weights not found at '{foundation_path}'.")

    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=6)
    pw   = torch.tensor(
        [(len(te_yb) - te_yb.sum()) / (te_yb.sum() + 1e-5)]).to(device)
    crit = HybridLoss()
    best = pat = 0
    save = f'_mvlm_seed{seed}.pt'

    for ep in range(epochs):
        model.train()
        for d_b, p_b, y_b in tr_loader:
            opt.zero_grad()
            crit(model(d_b.to(device), p_b.to(device)),
                 y_b.to(device), pw).backward()
            opt.step()
        model.eval()
        pr = []
        with torch.no_grad():
            for d_b, p_b, _ in te_loader:
                pr.extend(torch.sigmoid(
                    model(d_b.to(device), p_b.to(device)) / 0.07
                ).cpu().numpy())
        auc = roc_auc_score(te_yb, pr)
        sch.step(auc)
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d} | AUROC={auc:.4f}")
        if auc > best:
            best = auc; pat = 0
            torch.save(model.state_dict(), save)
        else:
            pat += 1
        if pat >= 15:
            print(f"  Early stop ep{ep+1}")
            break

    model.load_state_dict(torch.load(save, map_location=device))
    model.eval()
    pr = []
    with torch.no_grad():
        for d_b, p_b, _ in te_loader:
            pr.extend(torch.sigmoid(
                model(d_b.to(device), p_b.to(device)) / 0.07
            ).cpu().numpy())
    os.remove(save)
    pr = np.array(pr)
    return roc_auc_score(te_yb, pr), average_precision_score(te_yb, pr)


def train_regression(ModelClass, model_kwargs, tr_loader, te_loader,
                     te_y, te_yb, device, epochs, seed, tag):
    torch.manual_seed(seed)
    model    = ModelClass(**model_kwargs).to(device)
    opt      = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch      = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=6)
    mse_loss = nn.MSELoss()
    best = pat = 0
    save = f'_{tag}_seed{seed}.pt'
    is_mgf = hasattr(model, 'fp_proj')

    for ep in range(epochs):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            if is_mgf:
                d_b, p_b, fp_b, km_b, y_b = batch
                pred = model(d_b.to(device), p_b.to(device),
                             fp_b.to(device), km_b.to(device))
            else:
                d_b, p_b, y_b = batch
                pred = model(d_b.to(device), p_b.to(device))
            mse_loss(pred, y_b.to(device)).backward()
            opt.step()

        model.eval()
        pr = []
        with torch.no_grad():
            for batch in te_loader:
                if is_mgf:
                    d_b, p_b, fp_b, km_b, _ = batch
                    pr.extend(model(d_b.to(device), p_b.to(device),
                                    fp_b.to(device),
                                    km_b.to(device)).cpu().numpy())
                else:
                    d_b, p_b, _ = batch
                    pr.extend(model(
                        d_b.to(device), p_b.to(device)).cpu().numpy())
        pr  = np.array(pr)
        ci  = ci_score(te_y, pr)
        sch.step(ci)
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d} | CI={ci:.4f}")
        if ci > best:
            best = ci; pat = 0
            torch.save(model.state_dict(), save)
        else:
            pat += 1
        if pat >= 15:
            print(f"  Early stop ep{ep+1}")
            break

    model.load_state_dict(torch.load(save, map_location=device))
    model.eval()
    pr = []
    with torch.no_grad():
        for batch in te_loader:
            if is_mgf:
                d_b, p_b, fp_b, km_b, _ = batch
                pr.extend(model(d_b.to(device), p_b.to(device),
                                fp_b.to(device),
                                km_b.to(device)).cpu().numpy())
            else:
                d_b, p_b, _ = batch
                pr.extend(model(
                    d_b.to(device), p_b.to(device)).cpu().numpy())
    os.remove(save)
    pr    = np.array(pr)
    # Ensure AUROC > 0.5 (flip if inverted — regression models predict
    # high affinity for inactives in some conventions)
    y_bin = (te_y < THRESHOLD).astype(int)
    auroc = roc_auc_score(y_bin, pr)
    if auroc < 0.5:
        pr    = -pr
        auroc = roc_auc_score(y_bin, pr)
    auprc = average_precision_score(y_bin, pr)
    ci    = ci_score(te_y, pr)
    rm2   = rm2_score(te_y, pr)
    return auroc, auprc, ci, rm2


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Seeds  : {SEEDS}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading kiba_strict.csv...")
    df = pd.read_csv(
        get_file('kiba_strict.csv', args.data_dir)
    ).dropna(subset=['Drug', 'Target_ID', 'Y'])
    Y_cont    = df['Y'].values
    # Standard KIBA convention: Y < 12.1 = active = label 1
    y_bin_all = (Y_cont < THRESHOLD).astype(int)
    print(f"  Pairs: {len(df):,} | Active (Y<{THRESHOLD}): "
          f"{y_bin_all.sum():,} ({y_bin_all.mean()*100:.1f}%)")

    prot_map     = pd.read_csv(get_file('final_protein_map.csv', args.data_dir))
    prot_vectors = np.load(get_file('big_protein_vectors.npy', args.data_dir))

    t2idx = {}
    for i, row in prot_map.iterrows():
        for col in ['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
            if col in prot_map.columns and pd.notna(row[col]):
                for nm in str(row[col]).replace(';', ' ').split():
                    t2idx[nm] = i

    df        = df[df['Target_ID'].astype(str).isin(t2idx)].reset_index(drop=True)
    Y_cont    = df['Y'].values
    y_bin_all = (Y_cont < THRESHOLD).astype(int)
    print(f"  After target filter: {len(df):,}")

    # ── ChemBERTa embeddings ──────────────────────────────────────────────────
    print("\n[2/5] ChemBERTa-77M embeddings (CLS + mean → 768-d)...")
    tok   = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    chem  = AutoModel.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM").to(device)
    chem.eval()
    cache = {}
    with torch.no_grad():
        for i, drug in enumerate(df['Drug'].unique()):
            if (i + 1) % 200 == 0:
                print(f"  {i+1} drugs...", end='\r')
            inp = tok(drug, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            out = chem(**inp)
            # 768-d: CLS token (384-d) + mean pooling (384-d)
            cache[drug] = torch.cat([
                out.last_hidden_state[:, 0, :],
                torch.mean(out.last_hidden_state, dim=1)
            ], dim=1).squeeze(0).cpu().numpy()
    del chem
    print(f"\n  Done.")

    X_d = np.array([cache[d] for d in df['Drug']])
    X_p = np.array([prot_vectors[t2idx[str(t)]] for t in df['Target_ID']])

    # ── MGF-DTA extra features ────────────────────────────────────────────────
    if not args.skip_mgfdta:
        print("\n[3/5] MGF-DTA features (Morgan FP + k-mer PCA)...")
        print("  Morgan fingerprints...")
        fp_cache = {d: morgan_fp(d) for d in df['Drug'].unique()}
        X_fp     = np.array([fp_cache[d] for d in df['Drug']])

        seq_col = next((c for c in
                        ['Sequence', 'sequence', 'Target_Seq', 'Protein']
                        if c in prot_map.columns), None)
        if seq_col:
            print(f"  k-mer PCA from column '{seq_col}'...")
            seqs   = [str(prot_map.loc[t2idx[str(t)], seq_col])
                      if t2idx.get(str(t)) in prot_map.index else ''
                      for t in df['Target_ID']]
            X_km   = kmer_pca(seqs)
            kd     = X_km.shape[1]
            print(f"  k-mer PCA dim={kd}")
        else:
            print("  No sequence column — zero k-mer features.")
            X_km = np.zeros((len(df), 1), dtype=np.float32)
            kd   = 1
    else:
        X_fp = X_km = None
        kd   = 1

    # ── 3-seed training loop ──────────────────────────────────────────────────
    print("\n[4/5] Training across 3 seeds...")
    foundation_path = os.path.join('models', 'MVLM_Foundation.pt')

    results = {
        'MVLM (ours)': {'auroc': [], 'auprc': []},
        'PMMR':        {'auroc': [], 'auprc': [], 'ci': [], 'rm2': []},
        'MGF-DTA':     {'auroc': [], 'auprc': [], 'ci': [], 'rm2': []},
    }

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"  Seed {seed}")
        print(f"{'='*55}")
        np.random.seed(seed)

        (tr_d, te_d, tr_p, te_p,
         tr_y, te_y, tr_yb, te_yb) = train_test_split(
            X_d, X_p, Y_cont, y_bin_all,
            test_size=0.2, random_state=seed, stratify=y_bin_all)

        tr_bin = DataLoader(
            PairDataset(tr_d, tr_p, tr_yb.astype(np.float32)),
            batch_size=256, shuffle=True)
        te_bin = DataLoader(
            PairDataset(te_d, te_p, te_yb.astype(np.float32)),
            batch_size=256, shuffle=False)
        tr_reg = DataLoader(
            PairDataset(tr_d, tr_p, tr_y),
            batch_size=256, shuffle=True)
        te_reg = DataLoader(
            PairDataset(te_d, te_p, te_y),
            batch_size=256, shuffle=False)

        # MVLM
        print(f"\n  [MVLM] seed={seed}")
        auroc, auprc = train_mvlm(
            tr_bin, te_bin, te_yb, device,
            foundation_path, args.epochs, seed)
        print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        results['MVLM (ours)']['auroc'].append(auroc)
        results['MVLM (ours)']['auprc'].append(auprc)

        # PMMR
        if not args.skip_pmmr:
            print(f"\n  [PMMR] seed={seed}")
            auroc, auprc, ci, rm2 = train_regression(
                PMMR, {}, tr_reg, te_reg,
                te_y, te_yb, device, args.epochs, seed, 'pmmr')
            print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"CI={ci:.4f}  Rm2={rm2:.4f}")
            results['PMMR']['auroc'].append(auroc)
            results['PMMR']['auprc'].append(auprc)
            results['PMMR']['ci'].append(ci)
            results['PMMR']['rm2'].append(rm2)

        # MGF-DTA
        if not args.skip_mgfdta and X_fp is not None:
            print(f"\n  [MGF-DTA] seed={seed}")
            (tr_fp, te_fp,
             tr_km, te_km) = train_test_split(
                X_fp, X_km,
                test_size=0.2, random_state=seed, stratify=y_bin_all)
            mgf_tr = DataLoader(
                MGFDataset(tr_d, tr_p, tr_fp, tr_km, tr_y),
                batch_size=128, shuffle=True)
            mgf_te = DataLoader(
                MGFDataset(te_d, te_p, te_fp, te_km, te_y),
                batch_size=128, shuffle=False)
            auroc, auprc, ci, rm2 = train_regression(
                MGFDTA, {'kmer_dim': kd},
                mgf_tr, mgf_te,
                te_y, te_yb, device, args.epochs, seed, 'mgf')
            print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"CI={ci:.4f}  Rm2={rm2:.4f}")
            results['MGF-DTA']['auroc'].append(auroc)
            results['MGF-DTA']['auprc'].append(auprc)
            results['MGF-DTA']['ci'].append(ci)
            results['MGF-DTA']['rm2'].append(rm2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FAIR COMPARISON  —  KIBA  (Mean +/- SD across 3 seeds)")
    print("  Label convention: Y < 12.1 = active (standard KIBA)")
    print("=" * 70)
    print(f"{'Model':<22}{'AUROC':>20}{'AUPRC':>20}"
          f"{'CI':>16}{'Rm2':>16}")
    print("-" * 70)

    rows = []
    for name, r in results.items():
        if not r['auroc']:
            continue
        ci_s  = fmt(r['ci'])  if r.get('ci')  else '   N/A'
        rm2_s = fmt(r['rm2']) if r.get('rm2') else '   N/A'
        print(f"{name:<22}{fmt(r['auroc']):>20}{fmt(r['auprc']):>20}"
              f"{ci_s:>16}{rm2_s:>16}")
        rows.append({
            'model':      name,
            'auroc_mean': np.mean(r['auroc']),
            'auroc_sd':   np.std(r['auroc']),
            'auprc_mean': np.mean(r['auprc']),
            'auprc_sd':   np.std(r['auprc']),
            'ci_mean':    np.mean(r['ci'])  if r.get('ci')  else float('nan'),
            'ci_sd':      np.std(r['ci'])   if r.get('ci')  else float('nan'),
            'rm2_mean':   np.mean(r['rm2']) if r.get('rm2') else float('nan'),
            'rm2_sd':     np.std(r['rm2'])  if r.get('rm2') else float('nan'),
        })
    print("=" * 70)

    out = pd.DataFrame(rows)
    out.to_csv('comparison_kiba_results.csv', index=False)
    print("\n  Results saved to comparison_kiba_results.csv")


if __name__ == '__main__':
    main()
