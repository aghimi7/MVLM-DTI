"""
dataset.py

Provides PyTorch Dataset implementations for drug-target interaction data,
integrating on-the-fly tokenization and pre-computed protein vector mapping.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

class DTIDataset(Dataset):
    """
    Dataset loader for Drug-Target Interaction pairs.
    """
    def __init__(self, csv_path, protein_map_path, protein_vectors_path, target_col='label'):
        self.data = pd.read_csv(csv_path).dropna(subset=['Drug', 'Target_ID', target_col])
        self.protein_map = pd.read_csv(protein_map_path)
        self.protein_vectors = np.load(protein_vectors_path, allow_pickle=True)
        self.target_col = target_col
        
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        
        # Build protein index map across multiple potential identifier formats
        self.t2idx = {}
        for idx, row in self.protein_map.iterrows():
            for col in['Entry', 'Entry Name', 'Gene Names', 'Target_ID', 'Name']:
                if col in row and pd.notna(row[col]):
                    for n in str(row[col]).replace(';', ' ').split():
                        self.t2idx[n] = idx
                        
        # Filter data to only include targets present in the protein map
        self.data = self.data[self.data['Target_ID'].astype(str).isin(self.t2idx)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['Drug']
        target_id = str(row['Target_ID'])
        label = float(row[self.target_col])
        
        # Tokenize drug
        drug_tokens = self.tokenizer(
            smiles, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
        
        # Retrieve pre-computed ESM-2 protein embedding
        prot_idx = self.t2idx[target_id]
        prot_vec = torch.tensor(self.protein_vectors[prot_idx], dtype=torch.float32)

        return {
            'input_ids': drug_tokens['input_ids'].squeeze(0),
            'attention_mask': drug_tokens['attention_mask'].squeeze(0),
            'protein_embed': prot_vec,
            'label': torch.tensor(label, dtype=torch.float32)
        }