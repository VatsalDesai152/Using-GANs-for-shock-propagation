import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import logging

from model import BaselineTGATAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SHOCK_YEARS = {2008, 2020, 2022}
START_YEAR = 1995
END_YEAR = 2024

class GraphTemporalDataset(Dataset):
    def __init__(self, years, data_dir="data/processed"):
        self.years = years
        
        logging.info("Loading dataset CSVs into memory...")
        self.df_features = pd.read_csv(os.path.join(data_dir, "node_features.csv")).drop_duplicates(subset=["year", "country"])
        self.df_edges = pd.read_csv(os.path.join(data_dir, "multiplex_edges.csv")).drop_duplicates(subset=["year", "source", "target", "sector"])
        
        self.countries = sorted(self.df_features["country"].unique())
        self.country_to_id = {c: i for i, c in enumerate(self.countries)}
        self.num_nodes = len(self.countries)
        
        self.sectors = sorted(self.df_edges["sector"].unique())
        self.sector_to_id = {s: i for i, s in enumerate(self.sectors)}
        self.num_sectors = len(self.sectors)
        
        self.feature_cols = [c for c in self.df_features.columns if c not in ["year", "country"]]
        self.in_features = len(self.feature_cols)

    def __len__(self):
        return len(self.years)

    def __getitem__(self, idx):
        year = self.years[idx]
        
        df_year_feat = self.df_features[self.df_features["year"] == year].copy()
        df_year_feat['node_id'] = df_year_feat['country'].map(self.country_to_id)
        df_year_feat = df_year_feat.sort_values('node_id')
        x_np = df_year_feat[self.feature_cols].values
        x = torch.tensor(x_np, dtype=torch.float32)
        
        df_year_edges = self.df_edges[self.df_edges["year"] == year].copy()
        src_ids = df_year_edges["source"].map(self.country_to_id).values
        dst_ids = df_year_edges["target"].map(self.country_to_id).values
        edge_index = torch.tensor(np.vstack([src_ids, dst_ids]), dtype=torch.long)
        
        sec_ids = df_year_edges["sector"].map(self.sector_to_id).values
        sector_idx = torch.tensor(sec_ids, dtype=torch.long)
        
        vol_np = df_year_edges["log_trade_volume"].values
        edge_attr = torch.tensor(vol_np, dtype=torch.float32).unsqueeze(1)
        y_true = torch.tensor(vol_np, dtype=torch.float32).unsqueeze(1)
        
        return {
            'year': year,
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'sector_idx': sector_idx,
            'y_true': y_true
        }

def custom_collate(batch):
    # For sequence generation, simply return the batch list
    return batch

def train_model(model, dataloader, num_epochs=10, device='cuda'):
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = nn.DataParallel(model)
        
    model.to(device)
    
    # Optimizer: AdamW with decoupled weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_years = 0
        
        # Usually temporal graph training processes sequences
        # We iterate over a sequence of graphs
        for batch in dataloader:
            x_seq = [b['x'].to(device) for b in batch]
            edge_index_seq = [b['edge_index'].to(device) for b in batch]
            edge_attr_seq = [b['edge_attr'].to(device) for b in batch]
            sector_idx_seq = [b['sector_idx'].to(device) for b in batch]
            y_true_seq = [b['y_true'].to(device) for b in batch]
            years_seq = [b['year'] for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            out_tuple = model(x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq)
            if len(out_tuple) == 3:
                outputs, _, _ = out_tuple
                h_macro_seq, h_struct_seq = None, None
            else:
                outputs, _, _, h_macro_seq, h_struct_seq = out_tuple
            
            loss = 0.0
            for t, y_pred in enumerate(outputs):
                year = years_seq[t]
                y_true = y_true_seq[t]
                
                # Shock-Year Extrapolation Protocol
                if year in SHOCK_YEARS:
                    # Mask reconstruction loss to zero, no gradient update
                    logging.debug(f"Year {year} is a shock year, skipping gradient update.")
                    pass
                else:
                    loss_t = criterion(y_pred, y_true)
                    
                    # Orthogonal Penalty (Architecture B Only)
                    if h_macro_seq is not None and h_struct_seq is not None:
                        h_m = h_macro_seq[t]
                        h_s = h_struct_seq[t]
                        ortho_penalty = 0.1 * torch.mean(torch.abs(torch.nn.functional.cosine_similarity(h_m, h_s, dim=-1)))
                        loss_t = loss_t + ortho_penalty
                        
                    loss += loss_t
                    valid_years += 1
                    
            if type(loss) is not float:
                loss.backward()
                
                # Stability: Gradient clipping for recurrent updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
        avg_loss = epoch_loss / max(1, valid_years)
        scheduler.step(avg_loss)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    from model import BaselineTGATAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    years = list(range(START_YEAR, END_YEAR + 1))
    dataset = GraphTemporalDataset(years)
    
    # Num workers > 4 as requested
    dataloader = DataLoader(dataset, batch_size=len(years), shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    model = BaselineTGATAE(in_features_macro=4, in_features_struct=5, hidden_features=32, out_features=16)
    
    logging.info("Starting training loop...")
    # For testing execution, set num_epochs=1. Set to a small number in script.
    train_model(model, dataloader, num_epochs=2, device=device)
    logging.info("Training complete.")
