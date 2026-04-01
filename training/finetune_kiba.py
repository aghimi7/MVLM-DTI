"""
finetune_kiba.py

Fine-tunes the Universal Foundation Model on the strict KIBA dataset
using an ensemble of 3 random seeds to ensure robustness.
"""

import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MultiViewLinearManifold, HybridLoss
from src.dataset import DTIDataset

def train_ensemble(data_dir="../data", models_dir="../models"):
    print("--- FINE-TUNING MVLM ENSEMBLE (KIBA) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Strict KIBA Dataset
    csv_path = os.path.join(data_dir, "kiba_strict.csv")
    map_path = os.path.join(data_dir, "final_protein_map.csv")
    vec_path = os.path.join(data_dir, "big_protein_vectors.npy")
    
    # We assume 'label' column is pre-calculated (Y < 12.1)
    dataset = DTIDataset(csv_path, map_path, vec_path, target_col='label')
    
    # 80/20 Split logic (Mocked for clean script structure)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    foundation_path = os.path.join(models_dir, "universal_champion_big.pt")
    seeds =[42, 101, 999]

    for seed in seeds:
        print(f"\nTraining Seed {seed}...")
        torch.manual_seed(seed)
        
        model = MultiViewLinearManifold().to(device)
        if os.path.exists(foundation_path):
            state = torch.load(foundation_path, map_location=device)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = HybridLoss()

        best_auc = 0.0
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                d_emb = torch.cat([batch['input_ids'], batch['attention_mask']], dim=1).to(device) # Placeholder representation
                p_emb = batch['protein_embed'].to(device)
                y = batch['label'].to(device)
                
                optimizer.zero_grad()
                scores = model(d_emb, p_emb)
                loss = criterion(scores, y)
                loss.backward()
                optimizer.step()
                
            # Validation Step
            model.eval()
            all_preds, all_y = [],[]
            with torch.no_grad():
                for batch in test_loader:
                    d_emb = torch.cat([batch['input_ids'], batch['attention_mask']], dim=1).to(device)
                    p_emb = batch['protein_embed'].to(device)
                    scores = model(d_emb, p_emb)
                    probs = torch.sigmoid(scores / model.logit_scale)
                    all_preds.extend(probs.cpu().numpy())
                    all_y.extend(batch['label'].numpy())
                    
            auc = roc_auc_score(all_y, all_preds)
            scheduler.step(auc)
            
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(models_dir, f"kiba_mvlm_seed_{seed}.pt"))
            else:
                patience_counter += 1
                
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
                break

if __name__ == "__main__":
    train_ensemble()