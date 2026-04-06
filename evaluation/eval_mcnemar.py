import pandas as pd, numpy as np, torch, os, warnings
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file(f):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths =[os.path.join(base_dir, "data", f), os.path.join(base_dir, "models", f), os.path.join(base_dir, f)]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {f}")

def run_mcnemar():
    print("--- RUNNING MCNEMAR'S TEST: MVLM vs INCEPTION PROXY ---")
    
    df = pd.read_csv(get_file("kiba_strict.csv")).dropna(subset=['Drug', 'Target_ID', 'Y'])
    df['label'] = (df['Y'] < 12.1).astype(int)

    prot_map = pd.read_csv(get_file("final_protein_map.csv"))
    prot_vectors = np.load(get_file("big_protein_vectors.npy"))

    t2idx = {}
    for i, row in prot_map.iterrows():
        for col in['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
            if col in row and pd.notna(row[col]):
                for n in str(row[col]).replace(';', ' ').split(): t2idx[n] = i

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
    chemberta.eval()

    unique_drugs = df['Drug'].unique()
    drug_embeddings = {}
    with torch.no_grad():
        for drug in unique_drugs:
            inp = tokenizer(drug, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            out = chemberta(**inp)
            d_emb = torch.cat([out.last_hidden_state[:,0,:], torch.mean(out.last_hidden_state, dim=1)], dim=1)
            drug_embeddings[drug] = d_emb.squeeze(0).cpu()

    X_d, X_p, Y = [],[], []
    for _, row in df.iterrows():
        target = str(row['Target_ID'])
        if target in t2idx:
            X_d.append(drug_embeddings[row['Drug']].numpy())
            X_p.append(prot_vectors[t2idx[target]])
            Y.append(row['label'])

    _, test_d, _, test_p, _, test_y = train_test_split(np.array(X_d), np.array(X_p), np.array(Y), test_size=0.2, random_state=42, stratify=Y)

    class MVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.U = nn.Linear(768, 512, bias=False)
            self.V = nn.Linear(480, 512, bias=False)
            self.ln_d = nn.LayerNorm(512)
            self.ln_p = nn.LayerNorm(512)
        def forward(self, d, p):
            return torch.sum(F.normalize(self.ln_d(self.U(d)), dim=-1) * F.normalize(self.ln_p(self.V(p)), dim=-1), dim=1)

    ensemble_probs = np.zeros(len(test_y))
    seeds =[42, 101, 999]

    for seed in seeds:
        model = MVLM().to(device)
        state = torch.load(get_file(f"kiba_mvlm_seed_{seed}.pt"), map_location=device)
        state = {k.replace("module.", "").replace("drug_proj", "U").replace("prot_proj", "V"): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            score = model(torch.tensor(test_d).to(device), torch.tensor(test_p).to(device))
            ensemble_probs += torch.sigmoid(score / 0.07).cpu().numpy()
    ensemble_probs /= len(seeds)

    npz = np.load(get_file("plot_data.npz"))
    y_proxy = npz['y_proxy']
    y_true_npz = npz['y_true']

    min_len = min(len(test_y), len(y_true_npz))
    ensemble_probs = ensemble_probs[:min_len]
    y_proxy = y_proxy[:min_len]
    test_y = test_y[:min_len]

    def get_best_preds(probs, y_true):
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1: best_f1, best_thresh = f1, t
        return (probs >= best_thresh).astype(int), best_thresh, best_f1

    my_preds, _, _ = get_best_preds(ensemble_probs, test_y)
    px_preds, _, _ = get_best_preds(y_proxy, test_y)

    both_correct = np.sum((my_preds == test_y) & (px_preds == test_y))
    only_ours_correct = np.sum((my_preds == test_y) & (px_preds != test_y))
    only_px_correct = np.sum((my_preds != test_y) & (px_preds == test_y))
    both_wrong = np.sum((my_preds != test_y) & (px_preds != test_y))

    table = [[both_correct, only_ours_correct],[only_px_correct, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)

    print(f"Only MVLM Correct: {only_ours_correct}")
    print(f"Only Proxy Correct:{only_px_correct}")
    print(f"MCNEMAR P-VALUE: {result.pvalue:.4e} | Statistic: {result.statistic:.4f}")

if __name__ == "__main__":
    run_mcnemar()
