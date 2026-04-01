# MVLM-DTI: Multi-View Linear Manifold for Drug-Target Interaction

Official implementation of our manuscript detailing the **Multi-View Linear Manifold (MVLM)**. This repository provides the source code and inference engine to align pre-trained chemical and proteomic latent spaces for highly accurate, mathematically interpretable drug-target interaction prediction.

## Important Technical Notes for Users

Before running the model or processing your own data, please ensure your embeddings match the exact specifications of the pre-trained manifold:

*   **Protein Embeddings (ESM-2):** The model expects exactly **480-dimensional** vectors. You must use the 35M parameter version of ESM-2 (`esm2_t12_35M_UR50D`) and extract the representations from the final layer. Larger ESM models (e.g., 650M or 3B) will cause a dimension mismatch with the pre-trained projection matrices.
*   **Drug Embeddings (Multi-View ChemBERTa):** Small molecules are encoded using `DeepChem/ChemBERTa-77M-MLM`. To capture both global properties and local reactive substructures, the model requires a **768-dimensional Multi-View vector**. This is created by concatenating the global `[CLS]` token embedding (384 dims) with the mean-pooled embeddings of all atom tokens (384 dims).
*   **SMILES Canonicalization:** When using the Discovery Engine or preparing new datasets, ensure your input SMILES strings are valid. Under the hood, RDKit canonicalization is strongly recommended prior to tokenization to guarantee chemical and topological consistency with our Universal Foundation Set.

## Quick Start: The Discovery Engine

You can predict target kinases for any small molecule instantly. All required pre-trained weights and data mappings are included directly in this repository.

### 1. Installation
Requires Python 3.9+ and PyTorch 2.0+.
```bash
git clone https://github.com/aghimi7/MVLM-DTI.git
cd MVLM-DTI
pip install -r requirements.txt
```

### 2. Run the Inference Engine
Run the discovery engine script from your terminal:
```bash
python discovery_engine.py
```

*Example Output for Imatinib:*
```text
[Discovery Engine Validation]
Query SMILES: CC1=C(C=C(C=C1)NC(=O)C2=CC=C(...
Target ID       | Confidence
------------------------------
ABL1            | 0.9982
KIT             | 0.9845
PDGFRA          | 0.9712
```

## Repository Structure
*   `data/`: Contains structural maps, protein vectors, and strict benchmark splits.
*   `models/`: Contains the `.pt` checkpoints for the linear projection matrices.
*   `src/`: Core model definitions (MVLM architecture and Hybrid Loss).
*   `evaluation/`: Scripts to reproduce paper analyses (e.g., k-NN phylogeny, McNemar's test).
*   `data_processing/`: Scripts demonstrating strict cold-drug data splitting.

## Citation
If you use this code or our Universal Foundation Set in your research, please cite our paper:
```bibtex
[Citation Details Pending Publication]
```

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
