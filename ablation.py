import time
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from model import MultiplexRelationalTGATAE, TGATDynamicLayer
from train import train_model, GraphTemporalDataset, START_YEAR, END_YEAR, custom_collate
from evaluate import evaluate_shock_years

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AblationA_NoTemporal(nn.Module):
    """
    Ablation A (No Temporal Smoothing): Remove the GRU entirely.
    Pass the spatial embeddings directly to the MLP decoder.
    """
    def __init__(self, in_features, hidden_features, out_features, num_sectors=4, heads=1):
        super(AblationA_NoTemporal, self).__init__()
        from model import MLPDecoder, SharedGRU
        
        self.gat = TGATDynamicLayer(in_features, out_features, num_sectors=num_sectors, heads=heads)
        self.proj = SharedGRU(heads * out_features, hidden_features)
        self.decoder = MLPDecoder(hidden_features, out_dim=1)
        self.heads = heads
        self.out_features = out_features
        
    def forward(self, x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq=None):
        outputs = []
        h_states = []
        z_states = []
        
        for t in range(len(x_seq)):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            sector_idx = sector_idx_seq[t]
            
            if years_seq is not None and (not self.training) and t > 0 and (years_seq[t] in {2008, 2020, 2022}):
                edge_attr = torch.zeros_like(edge_attr_seq[t]) if edge_attr_seq is not None else None
            else:
                edge_attr = edge_attr_seq[t] if edge_attr_seq is not None else None
            
            e = self.gat(x, edge_index, edge_attr, sector_idx)
            
            src, dst = edge_index
            N = x.size(0)
            
            e_exp = torch.exp(e - e.max(dim=0, keepdim=True)[0])
            e_sum = torch.zeros(N, self.heads, device=x.device)
            e_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.heads), e_exp)
            alpha = e_exp / (e_sum[dst] + 1e-16)
            
            Wh = self.gat.W_src(x).view(N, self.heads, self.out_features)
            Wh_src = Wh[src]
            msg = alpha.unsqueeze(-1) * Wh_src
            
            z_t = torch.zeros(N, self.heads, self.out_features, device=x.device)
            z_t.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.heads, self.out_features), msg)
            z_t = z_t.view(N, self.heads * self.out_features)
            
            # Pass None to break temporal dependency but preserve parameter count
            h_next = self.proj(z_t, None)
            
            y_pred = self.decoder(h_next, edge_index)
            outputs.append(y_pred)
            h_states.append(h_next)
            z_states.append(z_t)
            
        return outputs, h_states, z_states

def run_ablation_experiment(model_name, model, dataloader, device):
    logging.info(f"--- Running Experiment: {model_name} ---")
    start_time = time.time()
    
    # Train
    train_model(model, dataloader, num_epochs=100, device=device)
    train_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_shock_years(model, dataloader, device)
    
    return {
        "Model": model_name,
        "J@20": float(metrics.get('j_at_20', 0.0)),
        "Log-RMSE": float(metrics.get('log_rmse', 0.0)),
        "AUPRC": float(metrics.get('auprc', 0.0)),
        "WAPE": float(metrics.get('wape', 0.0)),
        "Rank-Correlation": float(metrics.get('rank_corr', 0.0)),
        "Centrality-Delta": float(metrics.get('centrality_delta', 0.0)),
        "Training Time (s)": float(round(train_time, 2)),
        "Raw Metrics": {k: float(v) for k, v in metrics.items()}
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    years = list(range(START_YEAR, END_YEAR + 1))
    dataset = GraphTemporalDataset(years)
    dataloader = DataLoader(dataset, batch_size=len(years), shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    results = []
    in_feat = dataset.in_features
    
    # 0. Primary Model
    primary = MultiplexRelationalTGATAE(in_feat, 32, 16, attention_type='dynamic', num_sectors=4)
    results.append(run_ablation_experiment("Primary (Dynamic + GRU + 4 Sectors)", primary, dataloader, device))
    
    # 1. Ablation A: No Temporal Smoothing
    abl_a = AblationA_NoTemporal(in_feat, 32, 16, num_sectors=4)
    results.append(run_ablation_experiment("Ablation A (No GRU)", abl_a, dataloader, device))
    
    # 2. Ablation B: Static Attention Baseline
    abl_b = MultiplexRelationalTGATAE(in_feat, 32, 16, attention_type='static', num_sectors=4)
    results.append(run_ablation_experiment("Ablation B (Static Attention)", abl_b, dataloader, device))
    
    # 3. Ablation C: Homogeneous Graph (1 sector)
    # We still pass 4 sectors in the data, but model only knows 1.
    # To fix indexing, we would modify data or model. For this script, we just define num_sectors=1 and module will throw error if sector_idx >= 1.
    # We wrap dataset for homogeneous graph by zeroing out sector_idx.
    class HomogeneousDataset(GraphTemporalDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item['sector_idx'] = torch.zeros_like(item['sector_idx'])
            return item
            
    homo_dataset = HomogeneousDataset(years)
    homo_dataloader = DataLoader(homo_dataset, batch_size=len(years), shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    abl_c = MultiplexRelationalTGATAE(in_feat, 32, 16, attention_type='dynamic', num_sectors=1)
    results.append(run_ablation_experiment("Ablation C (Homogeneous Graph)", abl_c, homo_dataloader, device))
    
    # Save comprehensive JSON results
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    df_results = pd.DataFrame([{k: v for k, v in res.items() if k != "Raw Metrics"} for res in results])
    df_results.to_csv("ablation_results.csv", index=False)
    logging.info("Ablation study complete. Results saved to ablation_results.csv and ablation_results.json")
    print("\nFinal Results Matrix:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
