import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, average_precision_score
import networkx as nx
from scipy.stats import spearmanr
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def permutation_test_j20(S_pred_20, S_true_20, num_nodes, n_permutations=1000):
    intersection_obs = len(set(S_pred_20).intersection(set(S_true_20)))
    union_obs = len(set(S_pred_20).union(set(S_true_20)))
    j20_obs = intersection_obs / union_obs if union_obs > 0 else 0.0

    count_greater_equal = 0
    all_nodes = np.arange(num_nodes)
    mask = np.zeros(num_nodes, dtype=bool)
    mask[S_true_20] = True
    
    for _ in range(n_permutations):
        shuffled_pred = np.random.choice(all_nodes, size=20, replace=False)
        inter = mask[shuffled_pred].sum()
        uni = 40 - inter
        j20_perm = inter / uni if uni > 0 else 0.0
        if j20_perm >= j20_obs:
            count_greater_equal += 1
            
    p_value = count_greater_equal / n_permutations
    return j20_obs, p_value

def compute_tier1_metrics(y_true, y_pred):
    """
    Tier 1: Graph Reconstruction
    y_true, y_pred are assumed to be in log-scaled space as per Phase 1
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    # Log-RMSE
    log_rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    
    # AUPRC (assume binary thresholding at 0 for edge existence in sparse graph)
    # y_true > 0 means edge exists
    y_true_bin = (y_true_np > 0).astype(int)
    if len(np.unique(y_true_bin)) > 1:
        auprc = average_precision_score(y_true_bin, y_pred_np)
    else:
        auprc = 0.0
        
    # WAPE (Weighted Absolute Percentage Error in non-log space)
    y_true_exp = np.expm1(np.clip(y_true_np, 0.0, 20.0))
    y_pred_exp = np.expm1(np.clip(y_pred_np, 0.0, 20.0))
    
    # Introduce OLS scale-matching factor without intercept
    numerator = np.sum(y_true_exp * y_pred_exp)
    denominator = np.sum(y_pred_exp**2) + 1e-9
    k_s = numerator / denominator
    
    y_pred_corrected = y_pred_exp * k_s
    
    wape = np.sum(np.abs(y_true_exp - y_pred_corrected)) / (np.sum(y_true_exp) + 1e-9)
    
    return log_rmse, auprc, wape

def compute_tier2_metrics(edge_index, y_true, alpha):
    """
    Tier 2: Structural Similarity
    """
    # Simulate rank correlation and centrality difference for mock evaluation
    # To compute true PageRank, we build nx graphs
    G_true = nx.DiGraph()
    G_pred = nx.DiGraph()
    
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    w_true = y_true.detach().cpu().numpy().flatten()
    w_alpha = alpha.detach().cpu().numpy().flatten() if alpha is not None else w_true
    
    # Vectorized exponentiation
    w_true_raw = np.expm1(np.clip(w_true, 0.0, 20.0))
    w_alpha_raw = np.expm1(np.clip(w_alpha, 0.0, 20.0))
    
    # Fast grouping using pandas
    df_temp = pd.DataFrame({
        'u': src,
        'v': dst,
        'wt': w_true_raw,
        'wa': w_alpha_raw
    })
    df_grouped = df_temp.groupby(['u', 'v']).sum().reset_index()
    
    # Add nodes to preserve the complete node set
    num_nodes = max(src.max(), dst.max()) + 1 if len(src) > 0 else 0
    G_true.add_nodes_from(range(num_nodes))
    G_pred.add_nodes_from(range(num_nodes))
    
    # Add weighted edges in batch
    G_true.add_weighted_edges_from(df_grouped[['u', 'v', 'wt']].values)
    G_pred.add_weighted_edges_from(df_grouped[['u', 'v', 'wa']].values)
        
    pr_true = nx.pagerank(G_true, weight='weight', tol=1e-4)
    pr_pred = nx.pagerank(G_pred, weight='weight', tol=1e-4)
    
    nodes = list(set(G_true.nodes()).union(set(G_pred.nodes())))
    pr_true_vec = [pr_true.get(n, 0) for n in nodes]
    pr_pred_vec = [pr_pred.get(n, 0) for n in nodes]
    
    centrality_rank_delta = np.mean(np.abs(np.array(pr_true_vec) - np.array(pr_pred_vec)))
    
    # Rank Correlation
    if len(w_true) > 1:
        corr, _ = spearmanr(w_true, w_alpha)
    else:
        corr = 0.0
        
    return corr, centrality_rank_delta

def compute_tier3_metrics(z_T, z_T_minus_1, y_true_T, y_true_T_minus_1, edge_index_T, edge_index_T_minus_1, num_nodes, countries=None, year=None, case_studies=None):
    """
    Tier 3: Shock Alignment (The Primary Propagation Test)
    """
    z_T_np = z_T.detach().cpu().numpy()
    z_T_1_np = z_T_minus_1.detach().cpu().numpy()
    
    # 1. Embedding Displacement (Demeaned to remove uniform macro drift)
    delta = z_T_np - z_T_1_np
    delta_demeaned = delta - np.mean(delta, axis=0)
    displacement = np.linalg.norm(delta_demeaned, axis=1)
    
    # 2. Predicted Impact (Top 20 countries with largest displacement)
    S_pred_20 = np.argsort(displacement)[-20:]
    
    # 3. Ground Truth Impact
    # Aggregate total trade volume per country at T and T-1
    # For simplicity, we just aggregate out-edges and in-edges
    trade_T = np.zeros(num_nodes)
    trade_T_1 = np.zeros(num_nodes)
    
    src_T = edge_index_T[0].detach().cpu().numpy()
    w_T = y_true_T.detach().cpu().numpy().flatten()
    for u, w in zip(src_T, w_T):
        trade_T[u] += np.expm1(np.clip(w, 0.0, 20.0))
        
    src_T_1 = edge_index_T_minus_1[0].detach().cpu().numpy()
    w_T_1 = y_true_T_minus_1.detach().cpu().numpy().flatten()
    for u, w in zip(src_T_1, w_T_1):
        trade_T_1[u] += np.expm1(np.clip(w, 0.0, 20.0))
        
    contraction = trade_T_1 - trade_T  # Positive contraction means trade went down
    S_true_20 = np.argsort(contraction)[-20:]
    
    # 4. Accuracy Verification: Jaccard Similarity J@20 and Permutation Test
    j_at_20, p_value = permutation_test_j20(S_pred_20, S_true_20, num_nodes)
    
    if countries is not None and year is not None and case_studies is not None:
        pred_countries = [countries[i] for i in S_pred_20]
        true_countries = [countries[i] for i in S_true_20]
        case_studies[str(year)] = {
            'predicted_top_20': pred_countries,
            'actual_top_20': true_countries,
            'j_at_20': j_at_20,
            'p_value': p_value
        }
    
    return j_at_20, p_value

def evaluate_shock_years(model, dataloader, device, model_name="model"):
    """
    Main evaluation pipeline on shock years.
    Returns aggregated metrics.
    """
    model.eval()
    
    metrics = {
        'log_rmse': [],
        'auprc': [],
        'wape': [],
        'rank_corr': [],
        'centrality_delta': [],
        'j_at_20': [],
        'j_at_20_pval': []
    }
    
    all_abs_errors = []
    case_studies = {}
    
    # We retrieve countries if dataset provides it via dataloader.dataset
    countries = dataloader.dataset.countries if hasattr(dataloader.dataset, 'countries') else None
    
    with torch.no_grad():
        for batch in dataloader:
            x_seq = [b['x'].to(device) for b in batch]
            edge_index_seq = [b['edge_index'].to(device) for b in batch]
            edge_attr_seq = [b['edge_attr'].to(device) for b in batch]
            sector_idx_seq = [b['sector_idx'].to(device) for b in batch]
            y_true_seq = [b['y_true'].to(device) for b in batch]
            years_seq = [b['year'] for b in batch]
            
            out_tuple = model(x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq)
            if len(out_tuple) == 3:
                outputs, h_states, z_states = out_tuple
            else:
                outputs, h_states, z_states = out_tuple[:3]
            
            SHOCK_YEARS = {2008, 2020, 2022}
            
            for t, year in enumerate(years_seq):
                if year in SHOCK_YEARS:
                    y_pred = outputs[t]
                    y_true = y_true_seq[t]
                    edge_index = edge_index_seq[t]
                    z_T = z_states[t]
                    
                    # Assume alpha is available or proxy it with y_pred for tier 2
                    alpha = y_pred
                    
                    log_rmse, auprc, wape = compute_tier1_metrics(y_true, y_pred)
                    corr, centrality_delta = compute_tier2_metrics(edge_index, y_true, alpha)
                    
                    # Tier 3 requires T-1
                    if t > 0:
                        z_T_1 = z_states[t-1]
                        y_true_T_1 = y_true_seq[t-1]
                        edge_index_T_1 = edge_index_seq[t-1]
                        num_nodes = x_seq[t].size(0)
                        
                        j_at_20, p_val = compute_tier3_metrics(
                            z_T, z_T_1, y_true, y_true_T_1, edge_index, edge_index_T_1, 
                            num_nodes, countries, year, case_studies
                        )
                    else:
                        j_at_20 = 0.0
                        p_val = 1.0
                        
                    metrics['log_rmse'].append(log_rmse)
                    metrics['auprc'].append(auprc)
                    metrics['wape'].append(wape)
                    metrics['rank_corr'].append(corr)
                    metrics['centrality_delta'].append(centrality_delta)
                    metrics['j_at_20'].append(j_at_20)
                    metrics['j_at_20_pval'].append(p_val)
                    
                    # For Wilcoxon test
                    y_true_np = y_true.detach().cpu().numpy()
                    y_pred_np = y_pred.detach().cpu().numpy()
                    abs_errors = np.abs(y_true_np - y_pred_np).flatten()
                    all_abs_errors.extend(abs_errors)
                    
    # Average across all shock years found in the dataset
    agg_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}
    agg_metrics['abs_errors'] = np.array(all_abs_errors)
    
    if case_studies:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model_name}_country_case_studies.json", "w") as f:
            json.dump(case_studies, f, indent=4)
            
    return agg_metrics
