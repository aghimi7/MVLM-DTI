# Comparisons

This folder contains all comparison scripts used in the paper:

> *Interpretable Drug-Target Prediction via Geometric Alignment:
> A Linear Alternative to Deep Architectural Complexity*

---

## Scripts

### `compare_plm_methods.py`
Fair comparison of MVLM against two recently published PLM-based DTI methods:
- **PMMR** (Ouyang et al., Bioinformatics 2025)
- **MGF-DTA** (Ni et al., Int J Mol Sci 2026)

All three models are trained on identical splits of `kiba_strict.csv`
across three random seeds (42, 101, 999). Results are reported as Mean ± SD.

**Label convention:** KIBA score < 12.1 = active (label = 1)

```bash
python compare_plm_methods.py --data_dir /path/to/data --epochs 100
```

---

### `mcnemar_inception.py`
McNemar's test comparing MVLM against a faithful reimplementation of
InceptionDTA (Kalemati et al., Heliyon 2025) on the KIBA benchmark.

Both models are trained and evaluated on the same test split (seed=42).
The McNemar test assesses whether the difference in error patterns is
statistically significant.

```bash
python mcnemar_inception.py --data_dir /path/to/data --epochs 100
```

---

## Notes on official implementations

| Model | Official Code | Status |
|---|---|---|
| PMMR | https://github.com/NENUBioCompute/PMMR | Available (TF) — see `eval_pmmr_official.py` |
| MGF-DTA | https://github.com/fdmsz/MGF-DTA | Data files only, no model code |
| InceptionDTA | https://github.com/mz76m/InceptionDTA | TF 1.x — training instability on KIBA classification |

### PMMR official evaluation
The official PMMR TensorFlow implementation was evaluated on the same
test split (seed=42). Since PMMR's internal convention uses
Y >= 12.1 as positive (inactive class), predictions were post-hoc
corrected to the standard convention (Y < 12.1 = active):

```python
correct_labels = 1 - pmmr_labels
correct_probs  = 1 - pmmr_probs
```

This yielded **AUROC = 0.815, AUPRC = 0.939** for the official PMMR.

---

## Required data files

Place these files in the same directory or pass `--data_dir`:

- `kiba_strict.csv` — KIBA benchmark with contamination-removed drugs
- `final_protein_map.csv` — UniProt protein mapping
- `big_protein_vectors.npy` — ESM-2 protein embeddings
- `models/MVLM_Foundation.pt` — MVLM pretrained foundation weights

---

## Results reported in the paper (Table X)

| Model | AUROC ↑ | AUPRC ↑ |
|---|---|---|
| PMMR (official, seed=42) | 0.815 | 0.939 |
| MGF-DTA (proxy, seed=42) | 0.896 | 0.779 |
| **MVLM (ours, seed=42)** | **0.927** | **0.977** |
