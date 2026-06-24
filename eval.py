import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc
from torch.utils.data import DataLoader
import json

from model import BaselineTGATAE, OrthogonalTGATAE, DualStreamTGATAE
from models.baselines import OLSGravityModel
from train import GraphTemporalDataset, custom_collate, START_YEAR, END_YEAR
from evaluate import permutation_test_j20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SHOCK_YEARS = {2008, 2020, 2022}
SEEDS = list(range(42, 52))
NUM_SECTORS = 4

def evaluate_model(model, dataloader, device, model_name="model"):
    from evaluate import evaluate_shock_years
    return evaluate_shock_years(model, dataloader, device, model_name=model_name)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    years = list(range(START_YEAR, END_YEAR + 1))
    dataset = GraphTemporalDataset(years)
    dataloader = DataLoader(dataset, batch_size=len(years), shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    in_feat = dataset.in_features
    # Based on our data analysis, we have 4 macro and 5 structural features
    in_macro = 4
    in_struct = 5
    hidden_dim = 32
    out_dim = 16
    
    architectures = {
        'Architecture_A': BaselineTGATAE,
        'Architecture_B': OrthogonalTGATAE,
        'Architecture_C': DualStreamTGATAE,
        'OLS_Baseline': OLSGravityModel
    }
    
    all_results = {arch: [] for arch in architectures.keys()}
    
    for arch_name, ModelClass in architectures.items():
        logging.info(f"Evaluating {arch_name}...")
        for seed in SEEDS:
            if arch_name == 'OLS_Baseline':
                model = ModelClass()
            else:
                model = ModelClass(in_macro, in_struct, hidden_dim, out_dim, num_sectors=NUM_SECTORS)
            
            if hasattr(model, 'to'):
                model.to(device)
            else:
                # OLS model runs entirely on CPU anyway
                pass
            
            ckpt_path = f"results/multi_seed_models/{arch_name}_seed_{seed}.pt"
            
            if arch_name != 'OLS_Baseline':
                if os.path.exists(ckpt_path):
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                else:
                    logging.warning(f"Checkpoint not found: {ckpt_path}. Proceeding with untrained weights to demonstrate evaluation pipeline.")
            
            res = evaluate_model(model, dataloader, device, model_name=f"{arch_name}_seed_{seed}")
            all_results[arch_name].append(res)
            
    # Generate Markdown Table
    md = "# Grand Architectural Ablation Study Results\n\n"
    md += "| Architecture | Log-RMSE | AUPRC | WAPE | Rank-Correlation | Centrality-Delta | J@20 | J@20 P-Value |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    
    for arch_name in architectures.keys():
        df = pd.DataFrame(all_results[arch_name])
        if not df.empty:
            mean_rmse = df['log_rmse'].mean()
            std_rmse = df['log_rmse'].std()
            mean_auprc = df['auprc'].mean()
            std_auprc = df['auprc'].std()
            mean_wape = df['wape'].mean()
            std_wape = df['wape'].std()
            mean_rank_corr = df['rank_corr'].mean()
            std_rank_corr = df['rank_corr'].std()
            mean_centrality = df['centrality_delta'].mean()
            std_centrality = df['centrality_delta'].std()
            mean_j20 = df['j_at_20'].mean()
            std_j20 = df['j_at_20'].std()
            mean_pval = df['j_at_20_pval'].mean()
            std_pval = df['j_at_20_pval'].std()
            
            # Format avoiding NaNs for untrained runs
            std_rmse = 0.0 if np.isnan(std_rmse) else std_rmse
            std_auprc = 0.0 if np.isnan(std_auprc) else std_auprc
            std_wape = 0.0 if np.isnan(std_wape) else std_wape
            std_rank_corr = 0.0 if np.isnan(std_rank_corr) else std_rank_corr
            std_centrality = 0.0 if np.isnan(std_centrality) else std_centrality
            std_j20 = 0.0 if np.isnan(std_j20) else std_j20
            std_pval = 0.0 if np.isnan(std_pval) else std_pval
            
            md += f"| {arch_name} "
            md += f"| {mean_rmse:.4f} ± {std_rmse:.4f} "
            md += f"| {mean_auprc:.4f} ± {std_auprc:.4f} "
            md += f"| {mean_wape:.4f} ± {std_wape:.4f} "
            md += f"| {mean_rank_corr:.4f} ± {std_rank_corr:.4f} "
            md += f"| {mean_centrality:.4f} ± {std_centrality:.4f} "
            md += f"| {mean_j20:.4f} ± {std_j20:.4f} "
            md += f"| {mean_pval:.4f} ± {std_pval:.4f} |\n"
            
    print("\n" + "="*50 + "\n")
    print(md)
    print("="*50 + "\n")
    
    with open("eval_results_ablation.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    logging.info("Evaluation complete. Results written to eval_results_ablation.md")

if __name__ == "__main__":
    main()
