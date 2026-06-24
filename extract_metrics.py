import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from train import GraphTemporalDataset, custom_collate
from model import DualStreamTGATAE
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    years = list(range(1995, 2025))
    dataset = GraphTemporalDataset(years)
    dataloader = DataLoader(dataset, batch_size=len(years), shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    countries = dataset.countries
    country_to_id = {c: i for i, c in enumerate(countries)}
    
    target_countries = ['MEX', 'MLT', 'GBR', 'JPN', 'CHN', 'CAN', 'AUS', 'BRA', 'RUS', 'VEN', 'AFG']
    
    # 1. Compute Actual Trade Volume
    batch = next(iter(dataloader))
    years_seq = [b['year'] for b in batch]
    edge_index_seq = [b['edge_index'].to(device) for b in batch]
    y_true_seq = [b['y_true'].to(device) for b in batch]
    
    t_2019 = years_seq.index(2019)
    t_2020 = years_seq.index(2020)
    
    def get_trade_volume(t):
        trade = np.zeros(len(countries))
        src = edge_index_seq[t][0].cpu().numpy()
        dst = edge_index_seq[t][1].cpu().numpy()
        w = np.exp(np.clip(y_true_seq[t].cpu().numpy().flatten(), -20, 20)) - 1 # np.exp(log(1+y)) - 1
        # Aggregating out-edges and in-edges (as in previous logic)
        for u, v, weight in zip(src, dst, w):
            trade[u] += weight
            trade[v] += weight
        return trade

    vol_2019 = get_trade_volume(t_2019)
    vol_2020 = get_trade_volume(t_2020)
    
    decline_pct = ((vol_2020 - vol_2019) / (vol_2019 + 1e-9)) * 100
    
    # 2. Compute L2 Displacement for Architecture C
    in_macro = 4
    in_struct = 5
    hidden_dim = 32
    out_dim = 16
    NUM_SECTORS = 4
    
    SEEDS = list(range(42, 52))
    all_l2 = []
    
    x_seq = [b['x'].to(device) for b in batch]
    sector_idx_seq = [b['sector_idx'].to(device) for b in batch]
    edge_attr_seq = [b['edge_attr'].to(device) for b in batch]
    
    for seed in SEEDS:
        model = DualStreamTGATAE(in_macro, in_struct, hidden_dim, out_dim, num_sectors=NUM_SECTORS)
        ckpt_path = f"results/multi_seed_models/Architecture_C_seed_{seed}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs, h_states, z_states = model(x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq)
            
            z_2020 = z_states[t_2020].cpu().numpy()
            z_2019 = z_states[t_2019].cpu().numpy()
            
            delta = z_2020 - z_2019
            mu = np.mean(delta, axis=0)
            delta_debiased = delta - mu
            l2_shift = np.linalg.norm(delta_debiased, axis=1)
            all_l2.append(l2_shift)
            
    avg_l2 = np.mean(np.array(all_l2), axis=0)
    
    print("\n--- RESULTS ---")
    for c in target_countries:
        idx = country_to_id.get(c)
        if idx is not None:
            c_decline = decline_pct[idx]
            c_l2 = avg_l2[idx]
            print(f"- **{c}**: Actual Decline = {c_decline:.2f}%, Debiased L2 Displacement = {c_l2:.4f}")
        else:
            print(f"- **{c}**: Not found in dataset")

if __name__ == '__main__':
    main()
