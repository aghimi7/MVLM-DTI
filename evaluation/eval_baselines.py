import pandas as pd, numpy as np, torch, os, warnings
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

def get_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths =[os.path.join(base_dir, "data", filename)]
    for p in paths:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {filename}")

def run_concatenation_baselines():
    print("--- EVALUATING FEATURE CONCATENATION BASELINES ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(get_file("davis_strict.csv")).dropna(subset=['Drug', 'Target_ID', 'Y'])
    prot_map = pd.read_csv(get_file("final_protein_map.csv"))
    prot_vectors = np.load(get_file("big_protein_vectors.npy"))

    df['pKd'] = 9 - np.log10(df['Y'] + 1e-10)
    df['label'] = (df['pKd'] >= 7.0).astype(int)

    t2idx = {}
    for idx, row in prot_map.iterrows():
        for col in['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
            if col in row and pd.notna(row[col]):
                for n in str(row[col]).replace(';', ' ').split(): t2idx[n] = idx

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
            drug_embeddings[drug] = d_emb.squeeze(0).cpu().numpy()

    X, y = [], []
    for _, row in df.iterrows():
        target = str(row['Target_ID'])
        if target in t2idx:
            X.append(np.concatenate([drug_embeddings[row['Drug']], prot_vectors[t2idx[target]]]))
            y.append(row['label'])

    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        print("Error: No targets matched.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dataset compiled. Shape: {X.shape}. Training classifiers...")

    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_auc = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\n[Baseline Results]")
    print(f"Deep MLP (Concat):      {mlp_auc:.4f}")
    print(f"XGBoost (Concat):       {xgb_auc:.4f}")
    print(f"Random Forest (Concat): {rf_auc:.4f}")

if __name__ == "__main__":
    run_concatenation_baselines()
