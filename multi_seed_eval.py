import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import BaselineTGATAE, OrthogonalTGATAE, DualStreamTGATAE
from models.baselines import OLSGravityModel
from train import GraphTemporalDataset, custom_collate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Configuration
START_YEAR = 1995
END_YEAR = 2024
SHOCK_YEARS = {2008, 2020, 2022}
SEEDS = list(range(42, 52))  # 10 random seeds: 42 to 51
NUM_EPOCHS = 100
BATCH_SIZE = len(list(range(START_YEAR, END_YEAR + 1)))  # Full sequence batching
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ORTHOGONAL_LAMBDA = 0.1
NUM_SECTORS = 4

# Architecture mapping
ARCHITECTURES = {
    'Architecture_A': BaselineTGATAE,
    'Architecture_B': OrthogonalTGATAE,
    'Architecture_C': DualStreamTGATAE,
    'OLS_Baseline': OLSGravityModel
}

def set_reproducibility(seed):
    """Ensure fully reproducible weight initializations and batch shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_architecture(arch_name, ModelClass, seed, dataloader, device):
    """Trains a single architecture for a specific seed."""
    logging.info(f"Training {arch_name} - Seed {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    
    # Dataset features based on our pipeline
    in_macro = 4
    in_struct = 5
    hidden_dim = 32
    out_dim = 16
    
    # Instantiate Model
    if arch_name == 'OLS_Baseline':
        model = ModelClass()
    else:
        model = ModelClass(in_features_macro=in_macro, 
                           in_features_struct=in_struct, 
                           hidden_features=hidden_dim, 
                           out_features=out_dim, 
                           num_sectors=NUM_SECTORS).to(device)
    
    # Check if checkpoint exists
    os.makedirs("results/multi_seed_models", exist_ok=True)
    save_path = f"results/multi_seed_models/{arch_name}_seed_{seed}.pt"
    
    is_ols = (arch_name == 'OLS_Baseline')
    
    if not is_ols and os.path.exists(save_path):
        logging.info(f"Checkpoint {save_path} already exists. Skipping training for {arch_name} seed {seed}.")
        return

    # OLS is fit in a single shot inside its forward pass, no gradients needed
    if not is_ols:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        epochs = NUM_EPOCHS
    else:
        epochs = 1
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        if not is_ols:
            model.train()
        epoch_loss = 0.0
        valid_years = 0
        
        for batch in dataloader:
            x_seq = [b['x'].to(device) for b in batch]
            edge_index_seq = [b['edge_index'].to(device) for b in batch]
            edge_attr_seq = [b['edge_attr'].to(device) for b in batch]
            sector_idx_seq = [b['sector_idx'].to(device) for b in batch]
            y_true_seq = [b['y_true'].to(device) for b in batch]
            years_seq = [b['year'] for b in batch]
            
            if not is_ols:
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
                
                # Data Leakage Guard: Mask reconstruction loss for Shock Years
                if year in SHOCK_YEARS:
                    # Forward pass updates GRU state, but no backprop for this timestep
                    continue
                
                # Continuous MSE Loss
                loss_t = criterion(y_pred, y_true)
                
                # Architecture B: Dynamic Loss Function (Orthogonal Penalty)
                if h_macro_seq is not None and h_struct_seq is not None:
                    h_m = h_macro_seq[t]
                    h_s = h_struct_seq[t]
                    ortho_penalty = ORTHOGONAL_LAMBDA * torch.mean(torch.abs(F.cosine_similarity(h_m, h_s, dim=-1)))
                    loss_t = loss_t + ortho_penalty
                
                loss = loss + loss_t
                valid_years += 1
                
            if type(loss) is not float and valid_years > 0 and not is_ols:
                loss.backward()
                # Implement Gradient Clipping to prevent exploding gradients over the 30-year temporal sequence
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
        # Logging every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or is_ols:
            avg_loss = epoch_loss / max(1, valid_years)
            logging.info(f"[{arch_name} | Seed {seed}] Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")
            
    if not is_ols:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Successfully saved {arch_name} checkpoint for seed {seed} at {save_path}")
    
    # Memory management
    del model
    if not is_ols:
        del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    years = list(range(START_YEAR, END_YEAR + 1))
    dataset = GraphTemporalDataset(years)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    for arch_name, ModelClass in ARCHITECTURES.items():
        logging.info("=" * 60)
        logging.info(f"Starting Multi-Seed Training for {arch_name}")
        logging.info("=" * 60)
        
        for seed in SEEDS:
            set_reproducibility(seed)
            train_architecture(arch_name, ModelClass, seed, dataloader, device)
            
    logging.info("Grand Architectural Ablation Study Multi-Seed Training Complete!")

if __name__ == "__main__":
    main()
