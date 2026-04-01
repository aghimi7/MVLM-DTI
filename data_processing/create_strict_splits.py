"""
create_strict_splits.py

Removes data leakage between the Universal Foundation Set (BindingDB)
and downstream evaluation benchmarks (Davis, KIBA) by enforcing a 
strict cold-drug zero-overlap policy.
"""

import pandas as pd
import os

def generate_strict_benchmarks(data_dir="../data"):
    print("--- GENERATING STRICT ZERO-LEAKAGE BENCHMARKS ---")
    
    found_path = os.path.join(data_dir, "repaired_master_data.csv")
    davis_path = os.path.join(data_dir, "davis_repaired.csv")
    kiba_path = os.path.join(data_dir, "kiba_tdc_raw.csv")

    df_found = pd.read_csv(found_path)
    df_davis = pd.read_csv(davis_path)
    df_kiba = pd.read_csv(kiba_path)

    # Extract canonical representations
    found_drugs = set(df_found['smiles'].astype(str).str.strip().str.upper().dropna())

    def filter_and_save(df, name, out_filename):
        initial_len = len(df)
        df['smiles_clean'] = df['Drug'].astype(str).str.strip().str.upper()
        
        # Purge overlap
        df_strict = df[~df['smiles_clean'].isin(found_drugs)].copy()
        df_strict = df_strict.drop(columns=['smiles_clean'])
        
        final_len = len(df_strict)
        print(f"[{name}] Strict Filtering: {initial_len} -> {final_len} pairs. Removed {initial_len - final_len} overlapping compounds.")
        
        out_path = os.path.join(data_dir, out_filename)
        df_strict.to_csv(out_path, index=False)
        print(f"Saved strict dataset to {out_path}\n")

    filter_and_save(df_davis, "Davis", "davis_strict.csv")
    filter_and_save(df_kiba, "KIBA", "kiba_strict.csv")

if __name__ == "__main__":
    generate_strict_benchmarks()