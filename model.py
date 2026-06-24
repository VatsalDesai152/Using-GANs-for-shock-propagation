import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_gat import MultiplexMultiHeadGAT

class SharedGRU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SharedGRU, self).__init__()
        self.W_u = nn.Linear(hidden_features + in_features, hidden_features)
        self.W_r = nn.Linear(hidden_features + in_features, hidden_features)
        self.W_h = nn.Linear(hidden_features + in_features, hidden_features)
        
    def forward(self, z_t, h_prev):
        if h_prev is None:
            h_prev = torch.zeros(z_t.size(0), self.W_u.out_features, device=z_t.device)
            
        combined = torch.cat([h_prev, z_t], dim=-1)
        
        u = torch.sigmoid(self.W_u(combined))
        r = torch.sigmoid(self.W_r(combined))
        
        combined_reset = torch.cat([r * h_prev, z_t], dim=-1)
        h_cand = torch.tanh(self.W_h(combined_reset))
        
        h_next = (1 - u) * h_prev + u * h_cand
        return h_next

class MLPDecoder(nn.Module):
    def __init__(self, hidden_features, out_dim=1):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_dim)
        )
        
    def forward(self, h, edge_index):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        combined = torch.cat([h_src, h_dst], dim=-1)
        return self.net(combined)

class BaselineTGATAE(nn.Module):
    """Architecture A: Baseline T-GAT-AE (Coupled Temporal Fusion)"""
    def __init__(self, in_features_macro, in_features_struct, hidden_features, out_features, num_sectors=4, heads=1):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_features_macro + in_features_struct, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.gat = MultiplexMultiHeadGAT(hidden_features, out_features, num_sectors=num_sectors, heads=heads)
        self.gru = SharedGRU(heads * out_features, hidden_features)
        self.decoder = MLPDecoder(hidden_features, out_dim=1)
        
    def forward(self, x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq=None):
        h_prev = None
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
                
            x_fused = self.fusion_mlp(x)
            z_t = self.gat(x_fused, edge_index, sector_idx, edge_attr) # [N, heads * out_features]
            
            h_next = self.gru(z_t, h_prev)
            h_prev = h_next
            
            y_pred = self.decoder(h_next, edge_index)
            outputs.append(y_pred)
            h_states.append(h_next)
            z_states.append(z_t)
            
        return outputs, h_states, z_states

class OrthogonalTGATAE(nn.Module):
    """Architecture B: Orthogonal T-GAT-AE (Representation Disentanglement)"""
    def __init__(self, in_features_macro, in_features_struct, hidden_features, out_features, num_sectors=4, heads=1):
        super().__init__()
        self.macro_proj = nn.Linear(in_features_macro, hidden_features)
        self.struct_proj = nn.Linear(in_features_struct, hidden_features)
        
        self.gat = MultiplexMultiHeadGAT(2 * hidden_features, out_features, num_sectors=num_sectors, heads=heads)
        self.gru = SharedGRU(heads * out_features, hidden_features)
        self.decoder = MLPDecoder(hidden_features, out_dim=1)
        
    def forward(self, x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq=None):
        h_prev = None
        outputs = []
        h_states = []
        z_states = []
        h_macro_seq = []
        h_struct_seq = []
        
        in_macro = self.macro_proj.in_features
        for t in range(len(x_seq)):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            sector_idx = sector_idx_seq[t]
            
            if years_seq is not None and (not self.training) and t > 0 and (years_seq[t] in {2008, 2020, 2022}):
                edge_attr = torch.zeros_like(edge_attr_seq[t]) if edge_attr_seq is not None else None
            else:
                edge_attr = edge_attr_seq[t] if edge_attr_seq is not None else None
                
            x_macro = x[:, :in_macro]
            x_struct = x[:, in_macro:]
            
            h_macro = self.macro_proj(x_macro)
            h_struct = self.struct_proj(x_struct)
            
            x_fused = torch.cat([h_macro, h_struct], dim=-1)
            z_t = self.gat(x_fused, edge_index, sector_idx, edge_attr)
            
            h_next = self.gru(z_t, h_prev)
            h_prev = h_next
            
            y_pred = self.decoder(h_next, edge_index)
            
            outputs.append(y_pred)
            h_states.append(h_next)
            z_states.append(z_t)
            h_macro_seq.append(h_macro)
            h_struct_seq.append(h_struct)
            
        return outputs, h_states, z_states, h_macro_seq, h_struct_seq

class DualStreamTGATAE(nn.Module):
    """Architecture C: Dual-Stream T-GAT-AE (Decoupled Temporal Fusion)"""
    def __init__(self, in_features_macro, in_features_struct, hidden_features, out_features, num_sectors=4, heads=1):
        super().__init__()
        self.macro_proj = nn.Linear(in_features_macro, hidden_features)
        self.struct_proj = nn.Linear(in_features_struct, hidden_features)
        
        self.gru_macro = SharedGRU(hidden_features, hidden_features)
        
        self.gat = MultiplexMultiHeadGAT(hidden_features, out_features, num_sectors=num_sectors, heads=heads)
        self.gru_topo = SharedGRU(heads * out_features, hidden_features)
        
        # Late Fusion Decoder
        self.decoder = MLPDecoder(2 * hidden_features, out_dim=1)
        
    def forward(self, x_seq, edge_index_seq, edge_attr_seq, sector_idx_seq, years_seq=None):
        h_prev_macro = None
        h_prev_topo = None
        outputs = []
        h_states = []
        z_states = [] # This will be gru_topo states for Architecture C
        
        in_macro = self.macro_proj.in_features
        for t in range(len(x_seq)):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            sector_idx = sector_idx_seq[t]
            
            if years_seq is not None and (not self.training) and t > 0 and (years_seq[t] in {2008, 2020, 2022}):
                edge_attr = torch.zeros_like(edge_attr_seq[t]) if edge_attr_seq is not None else None
            else:
                edge_attr = edge_attr_seq[t] if edge_attr_seq is not None else None
                
            x_macro = x[:, :in_macro]
            x_struct = x[:, in_macro:]
            
            h_macro = self.macro_proj(x_macro)
            h_struct = self.struct_proj(x_struct)
            
            h_next_macro = self.gru_macro(h_macro, h_prev_macro)
            h_prev_macro = h_next_macro
            
            z_t = self.gat(h_struct, edge_index, sector_idx, edge_attr)
            
            h_next_topo = self.gru_topo(z_t, h_prev_topo)
            h_prev_topo = h_next_topo
            
            # Late Fusion
            h_fused = torch.cat([h_next_macro, h_next_topo], dim=-1)
            y_pred = self.decoder(h_fused, edge_index)
            
            outputs.append(y_pred)
            h_states.append(h_fused)
            # Extracted topological state for evaluation (as requested by prompt)
            z_states.append(h_next_topo)
            
        return outputs, h_states, z_states

