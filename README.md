# T-GAT-AE: The Limits of Topology in Trade Network Shock Propagation

> **Volume Prediction vs. Structural Reconfiguration in Spatiotemporal Trade Network Models**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository implements a **Multiplex Relational Temporal Graph Attention Autoencoder (T-GAT-AE)** that models international trade as a dynamic, multiplex graph spanning 30 years (1995–2024). The project investigates a fundamental question in applied machine learning:

> *Does the choice of attention mechanism in Graph Attention Networks affect the model's ability to predict how economic shocks propagate through international trade networks?*

### Central Finding — "The Limits of Topology"

Our experiments reveal a fundamental **decoupling** between volume prediction and topology reconstruction:

| Task | Winner | Metric |
|------|--------|--------|
| **Aggregate volume prediction** | OLS Gravity Baseline | WAPE = 1.658 |
| **Topology reconstruction** | Architecture C (Dual-Stream T-GAT-AE) | AUPRC = 0.846 |
| **Shock identification consistency** | Architecture C | J@20 = 0.051 ± 0.002 |

Linear models excel at predicting *how much* trade occurs; graph neural networks excel at predicting *which links* survive during a crisis. This divergence establishes "the limits of topology" — the point at which structural complexity ceases to improve aggregate prediction.

### Shock Events Studied

| Year | Event | Type |
|------|-------|------|
| **2008** | Global Financial Crisis | Financial contagion |
| **2020** | COVID-19 Pandemic | Supply chain disruption |
| **2022** | Russia-Ukraine Conflict | Geopolitical energy shock |

---

## Architecture

The project compares four models across a rigorous **shock-year holdout protocol** where crisis years are withheld from gradient updates:

### Architecture A: Baseline T-GAT-AE (Coupled Temporal Fusion)
Fuses all 9 features via MLP → MultiplexGAT → SharedGRU → Decoder

### Architecture B: Orthogonal T-GAT-AE (Representation Disentanglement)
Separately projects macro (4-dim) and structural (5-dim) features with an orthogonal penalty to encourage disentanglement.

### Architecture C: Dual-Stream T-GAT-AE (Decoupled Temporal Fusion)
Fully decoupled pipelines — macro features pass through their own GRU; structural features pass through GAT + separate GRU. Late fusion at decoding.

### OLS Structural Gravity Baseline
A traditional linear gravity model using Ridge regression on concatenated node features.

```
Architecture C (Best Topology Reconstruction):

  ┌─────────────────┐
  │  9-Dim Input xᵢᵗ │
  │  (4 Macro + 5    │
  │   Structural)    │
  └────────┬────────┘
           │ Split
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ Macro   │  │ Struct   │
│ Project │  │ Project  │
└────┬────┘  └────┬─────┘
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ GRU     │  │ Multiplex    │
│ (Macro) │  │ Multi-Head   │
└────┬────┘  │ GAT (GATv2)  │
     │       └──────┬───────┘
     │              ▼
     │       ┌──────────┐
     │       │ GRU      │
     │       │ (Topo)   │
     │       └────┬─────┘
     └──────┬─────┘
            ▼
     ┌──────────────┐
     │  Late Fusion  │
     │  [hᵐ ∥ hᵗ]   │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │ Neural Gravity│
     │ Decoder (MLP) │
     └──────────────┘
```

---

## Key Innovation: Sector-Specific Attention Isolation

The **MultiplexMultiHeadGAT** shares all linear projections (W_src, W_dst, W_edge, W_val) across all four commodity sectors. The **only** sector-specific parameter is the attention query vector `a_s`. This guarantees that any predictive disparity between sectors (Food, Energy, Semiconductors, Consumer Goods) is attributable solely to how the attention mechanism *routes information* — not to differences in feature encoding capacity.

---

## Repository Structure

```
├── data_pipeline.py           # Multi-source data ingestion (5 concurrent threads)
├── model.py                   # Three T-GAT-AE architectures (A, B, C)
├── train.py                   # Training loop + dataset construction
├── evaluate.py                # Three-tier evaluation framework
├── eval.py                    # Grand architectural ablation evaluation
├── multi_seed_eval.py         # 10-seed reproducible training (A/B/C + OLS)
├── ablation.py                # Component ablation study
├── ppml_baseline.py           # PPML Gravity baseline (Silva & Tenreyro, 2006)
├── volume_conditioned_metric.py  # Volume-conditioned displacement metric
├── extract_metrics.py         # Country-level metric extraction
├── generate_figures.py        # Publication-quality figure generation (6 figures)
├── generate_docx.py           # EAAI/Elsevier DOCX manuscript generator
├── generate_latex.py          # AIAA LaTeX paper generator
├── generate_latex_aer.py      # AER LaTeX paper generator
├── orchestrate.py             # End-to-end pipeline orchestrator
├── requirements.txt           # Python dependencies
│
├── models/
│   ├── __init__.py
│   ├── temporal_gat.py        # MultiplexMultiHeadGAT (GATv2 + sector-specific attn)
│   ├── cross_transformer.py   # CrossSectorTransformer (experimental)
│   └── baselines.py           # Node2Vec, DeepWalk, GAE, OLS Gravity
│
├── data/
│   ├── raw/                   # Source datasets (user-provided, not tracked in git)
│   │   ├── BACI_HS92_V202601.zip      # CEPII bilateral trade (~2.4 GB)
│   │   ├── GEDEvent_v26_0_4.csv       # UCDP conflict events
│   │   ├── country_mapping.parquet    # BACI country code → ISO3
│   │   └── dist_cepii.xls            # CEPII gravity variables
│   └── processed/             # Pipeline outputs (generated by data_pipeline.py)
│       ├── node_features.csv          # Country×year feature matrix
│       └── multiplex_edges.csv        # Directed trade edges with sectors
│
├── results/                   # Evaluation outputs and model checkpoints
│   ├── multi_seed_models/     # 80 .pt checkpoints (8 models × 10 seeds)
│   └── *.json                 # Evaluation results
│
├── figures/                   # Publication figures (PDF + PNG at 300 DPI)
│   ├── fig1_dual_stream_architecture.*
│   ├── fig2_shock_year_protocol.*
│   ├── fig3_wape_vs_auprc.*
│   ├── fig4_pr_curves_2020.*
│   ├── fig5_displacement_vs_decline.*
│   └── fig6_null_distribution_j20.*
│
└── output/                    # Generated manuscripts
    ├── Econ-Paper-Draft1-AIAA.tex
    ├── Econ-Paper-Draft1-AER.tex
    └── Econ_Paper_v3_Formatted.docx
```

---

## Data Pipeline

The pipeline ingests five heterogeneous data sources via concurrent threads:

| Source | Type | Variables | Access |
|--------|------|-----------|--------|
| **BACI** (CEPII) | Bilateral trade | Trade volumes by HS6 code → 4 sectors | Bulk ZIP |
| **FRED** (St. Louis Fed) | US macro | Fed funds, CPI, USD index, unemployment | REST API |
| **OWID** (Oxford) | Development | GDP, population, fossil energy share | Python library |
| **FAOSTAT** (FAO) | Agriculture | Wheat, rice, maize, soybean production | REST API |
| **UCDP** (Uppsala) | Conflict | Georeferenced conflict fatalities | Bulk CSV |

### Sector Classification

| Sector | HS Codes | Description |
|--------|----------|-------------|
| Food | Chapters 01–24 | Agricultural products, food preparations |
| Energy | Chapter 27 | Mineral fuels, petroleum, natural gas |
| Semiconductors | 8541, 8542 | Diodes, transistors, integrated circuits |
| Consumer Goods | All other | Residual (vehicles, textiles, machinery) |

### Feature Engineering

Each country-year is represented by a **9-dimensional** feature vector:
- **4 Global Macro** (broadcast to all countries): fed funds rate, CPI, USD index, unemployment
- **5 Country-Specific Structural**: GDP growth, population growth, fossil energy share, food production, conflict fatalities

Features are log-differenced for stationarity and StandardScaler-normalized (fit on non-shock years only to prevent data leakage).

---

## Evaluation Framework

All metrics are computed **exclusively on held-out shock years** (2008, 2020, 2022):

### Tier 1: Predictive Accuracy
| Metric | Description | Direction |
|--------|-------------|-----------|
| **Log-RMSE** | RMSE in log-space (handles scale variation) | ↓ Lower is better |
| **AUPRC** | Edge existence prediction under class imbalance | ↑ Higher is better |
| **WAPE** | Weighted absolute percentage error on trade levels | ↓ Lower is better |

### Tier 2: Structural Similarity
| Metric | Description | Direction |
|--------|-------------|-----------|
| **Rank Correlation** (Spearman ρ) | Attention weights vs. import dependency | ↑ Higher is better |
| **Centrality Rank Delta** | PageRank difference between true and learned graphs | ↓ Lower is better |

### Tier 3: Shock Alignment
| Metric | Description | Direction |
|--------|-------------|-----------|
| **J@20** (Jaccard@20) | Overlap between predicted and actual top-20 impacted countries | ↑ Higher is better |
| **Permutation p-value** | Statistical significance (10,000 permutations) | ↓ Lower is better |

---

## Results

### Grand Architectural Comparison (N = 10 seeds)

| Architecture | Log-RMSE ↓ | AUPRC ↑ | WAPE ↓ | Rank-Corr ↑ | J@20 ↑ |
|---|---|---|---|---|---|
| OLS Baseline | 4.301 ± 0.000 | 0.567 ± 0.000 | **1.658 ± 0.000** | −0.024 ± 0.000 | 0.129 ± 0.000 |
| Arch A (Coupled) | 3.964 ± 0.259 | 0.753 ± 0.094 | 1.912 ± 0.182 | 0.473 ± 0.153 | 0.073 ± 0.055 |
| Arch B (Orthogonal) | 4.046 ± 0.457 | 0.767 ± 0.084 | 1.889 ± 0.342 | 0.494 ± 0.152 | 0.042 ± 0.042 |
| **Arch C (Dual-Stream)** | **3.722 ± 0.596** | **0.846 ± 0.040** | 1.742 ± 0.319 | **0.627 ± 0.072** | 0.079 ± 0.074 |

### Key Findings

1. **OLS dominates volume prediction** (WAPE) — linear gravity models are hard to beat for aggregate forecasting
2. **Architecture C dominates topology reconstruction** (AUPRC = 0.846) — decoupled processing prevents macro noise from contaminating structural learning
3. **Architecture C has the best economic interpretability** (Rank-Corr = 0.627) — attention weights align with real import dependencies
4. **Dynamic vs. static attention produces significantly different errors** (Wilcoxon p < 0.0001)
5. **GRU temporal smoothing is essential** — removing it causes catastrophic instability

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, not required)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/T-GAT-AE.git
cd T-GAT-AE
pip install -r requirements.txt
```

### Data Preparation

The pipeline requires several external datasets. Place them in `data/raw/`:

1. **BACI Trade Data**: Download from [CEPII](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) → `data/raw/BACI_HS92_V202601.zip`
2. **UCDP Conflict Data**: Download from [UCDP](https://ucdp.uu.se/downloads/) → `data/raw/GEDEvent_v26_0_4.csv`
3. **CEPII Distance Data**: Download from [CEPII](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6) → `data/raw/dist_cepii.xls`
4. **FRED API Key**: Register at [FRED](https://fredaccount.stlouisfed.org/apikeys) and set in `data_pipeline.py`

### Running the Full Pipeline

```bash
# Step 1: Build the data pipeline (fetches FRED, OWID, FAOSTAT; processes BACI + UCDP)
python data_pipeline.py

# Step 2: Train all architectures across 10 random seeds
python multi_seed_eval.py

# Step 3: Evaluate all architectures on shock years
python eval.py

# Step 4: Generate publication figures
python generate_figures.py

# Or run everything end-to-end:
python orchestrate.py
```

### Quick Test (No Data Required)

To verify the model architecture works:

```python
import torch
from model import DualStreamTGATAE

model = DualStreamTGATAE(
    in_features_macro=4,
    in_features_struct=5,
    hidden_features=32,
    out_features=16,
    num_sectors=4
)

# Simulate 3-year sequence with 10 countries
x_seq = [torch.randn(10, 9) for _ in range(3)]
edge_index_seq = [torch.randint(0, 10, (2, 20)) for _ in range(3)]
edge_attr_seq = [torch.randn(20, 1) for _ in range(3)]
sector_idx_seq = [torch.randint(0, 4, (20,)) for _ in range(3)]

outputs, h_states, z_states = model(x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq)
print(f"Predictions shape: {outputs[0].shape}")  # [20, 1]
```

---

## Shock-Year Holdout Protocol

The most critical aspect of the experimental design:

- **During training**: Shock years (2008, 2020, 2022) undergo a **forward pass only** — the GRU state is updated for temporal continuity, but reconstruction losses are **masked to zero** (no gradients)
- **During inference**: Edge attributes for shock years are **zeroed out**, simulating the model having no information about the shock's magnitude
- **Effect**: Transforms the evaluation from interpolation to genuine **extrapolation**

---

## Publication Outputs

The repository includes scripts to generate publication-ready outputs:

| Script | Output | Format |
|--------|--------|--------|
| `generate_figures.py` | 6 publication figures | PDF (vector) + PNG (300 DPI) |
| `generate_latex.py` | AIAA conference paper | LaTeX (.tex) |
| `generate_latex_aer.py` | AER journal paper | LaTeX (.tex) |
| `generate_docx.py` | Elsevier/EAAI manuscript | Word (.docx) |

All figures use color-blind friendly palettes (Seaborn 'colorblind' + viridis) and comply with Elsevier formatting standards.

---

## Technical Notes

### Why Pure Python Scatter (No PyG C++ Kernels)?

The `MultiplexMultiHeadGAT` implements a custom `_safe_softmax` using `torch.scatter_add_` and `scatter_reduce_` rather than PyTorch Geometric's native C++ scatter kernels. This avoids **Access Violation (0xC0000005)** crashes on Windows with certain graph sizes. All baseline models (Node2Vec, DeepWalk, GAE) are similarly implemented in pure Python/PyTorch.

### Why GRU over LSTM/Transformer?

With only 30 annual observations, transformers and LSTMs risk overfitting. Trade relationships exhibit strong temporal inertia — the GRU's gating naturally captures the balance between long-run persistence and short-run structural disruption.

---

## References

1. Baldwin, R. (2009). "The Great Trade Collapse." *VoxEU.org*.
2. Javorcik, B. (2020). "Global supply chains will not be the same in the post-COVID-19 world."
3. Ruta, M. (2022). "The Impact of the War in Ukraine on Global Trade and Investment." *World Bank*.
4. Anderson, J. E., & van Wincoop, E. (2003). "Gravity with Gravitas." *AER*, 93(1): 170–192.
5. Acemoglu, D. et al. (2012). "The Network Origins of Aggregate Fluctuations." *Econometrica*, 80(5): 1977–2016.
6. Veličković, P. et al. (2018). "Graph Attention Networks." *ICLR*.
7. Brody, S. et al. (2022). "How Attentive are Graph Attention Networks?" *ICLR*.
8. Silva, J.M.C.S., & Tenreyro, S. (2006). "The Log of Gravity." *Review of Economics and Statistics*, 88(4): 641–658.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{desai2026limits,
  title={The Limits of Topology: Volume Prediction vs. Structural
         Reconfiguration in Spatiotemporal Trade Network Models},
  author={Desai, Vatsal},
  year={2026}
}
```
