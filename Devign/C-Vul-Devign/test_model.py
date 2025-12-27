"""Test vulnerability detection on C code."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")


def load_threshold() -> float:
    """Load threshold from ensemble_config.json."""
    if ENSEMBLE_CONFIG_PATH.exists():
        config = json.loads(ENSEMBLE_CONFIG_PATH.read_text())
        return float(config.get("optimal_threshold", 0.65))
    return 0.65

class HierarchicalBiGRU(nn.Module):
    def __init__(self, vocab_size=238, embed_dim=96, hidden_dim=192, slice_hidden=160,
                 vuln_dim=26, slice_feat_dim=52, gate_init=0.3):
        super().__init__()
        self.slice_hidden = slice_hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.3)
        
        self.global_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)
        self.global_attn = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        
        self.slice_gru = nn.GRU(embed_dim, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.slice_seq_gru = nn.GRU(slice_hidden*2, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_seq_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.slice_feat_mlp = nn.Sequential(nn.Linear(slice_feat_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4))
        self.slice_fusion = nn.Sequential(
            nn.Linear(slice_hidden*2 + 128, slice_hidden*2),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.slice_level_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.vuln_dim = vuln_dim
        self.vuln_mlp = nn.Sequential(nn.BatchNorm1d(vuln_dim), nn.Linear(vuln_dim, 64), nn.GELU(), nn.Dropout(0.4))
        
        self.feature_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.gate_strength_raw = nn.Parameter(torch.tensor(gate_init))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim*2 + slice_hidden*2 + 64), 
            nn.Linear(hidden_dim*2 + slice_hidden*2 + 64, 256), 
            nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    @property
    def gate_strength(self):
        return torch.sigmoid(self.gate_strength_raw)
    
    def encode_global(self, ids, mask):
        emb = self.embed_drop(self.embedding(ids))
        out, _ = self.global_gru(emb)
        scores = self.global_attn(out).masked_fill(mask.unsqueeze(-1)==0, -65000.0)
        return (out * F.softmax(scores, dim=1)).sum(dim=1)
    
    def encode_slices(self, slice_ids, slice_mask, slice_count, slice_vuln=None, slice_rel=None):
        B, S, L = slice_ids.shape
        emb = self.embed_drop(self.embedding(slice_ids.view(B*S, L)))
        out, _ = self.slice_gru(emb)
        scores = self.slice_attn(out).masked_fill(slice_mask.view(B*S,L).unsqueeze(-1)==0, -65000.0)
        slice_repr = (out * F.softmax(scores, dim=1)).sum(dim=1).view(B, S, -1)
        
        if slice_vuln is not None and slice_rel is not None:
            feat = self.slice_feat_mlp(torch.cat([slice_vuln, slice_rel], dim=-1))
            slice_repr = self.slice_fusion(torch.cat([slice_repr, feat], dim=-1))
        
        s_mask = torch.arange(S, device=slice_count.device).expand(B,S) < slice_count.unsqueeze(1)
        
        s_scores = self.slice_level_attn(slice_repr).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_attn_repr = (slice_repr * F.softmax(s_scores, dim=1)).sum(dim=1)
        
        slice_repr_masked = slice_repr * s_mask.unsqueeze(-1).float()
        seq_out, _ = self.slice_seq_gru(slice_repr_masked)
        seq_scores = self.slice_seq_attn(seq_out).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_seq_repr = (seq_out * F.softmax(seq_scores, dim=1)).sum(dim=1)
        
        return 0.5 * (slice_attn_repr + slice_seq_repr)
    
    def forward(self, input_ids, attention_mask, slice_input_ids=None, slice_attention_mask=None, 
                slice_count=None, vuln_features=None, slice_vuln_features=None, slice_rel_features=None, **kw):
        g = self.encode_global(input_ids, attention_mask)
        s = self.encode_slices(slice_input_ids, slice_attention_mask, slice_count, slice_vuln_features, slice_rel_features) if slice_input_ids is not None else torch.zeros(g.size(0), self.slice_hidden*2, device=g.device)
        
        if vuln_features is not None:
            v = self.vuln_mlp(vuln_features)
            gate = self.feature_gate(v)
            v = v * (1.0 + self.gate_strength * (gate - 0.5))
        else:
            v = torch.zeros(g.size(0), 64, device=g.device)
        
        h = torch.cat([g, s, v], dim=1)
        logits = self.classifier(h)
        
        return logits


def load_model(model_path):
    model = HierarchicalBiGRU()
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def test_dummy():
    """Test with dummy input to verify model works."""
    model = load_model('Output/output/models/best_v2_seed42.pt')
    print("Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    B = 1  # batch size
    input_ids = torch.randint(0, 238, (B, 512))
    attention_mask = torch.ones(B, 512)
    slice_input_ids = torch.randint(0, 238, (B, 8, 256))
    slice_attention_mask = torch.ones(B, 8, 256)
    slice_count = torch.tensor([4])
    vuln_features = torch.randn(B, 26)
    slice_vuln_features = torch.randn(B, 8, 26)
    slice_rel_features = torch.randn(B, 8, 26)
    
    with torch.no_grad():
        logits = model(
            input_ids, attention_mask,
            slice_input_ids, slice_attention_mask,
            slice_count, vuln_features,
            slice_vuln_features, slice_rel_features
        )
        probs = F.softmax(logits, dim=1)
        vuln_prob = probs[0, 1].item()
        
    threshold = load_threshold()
    print(f"\nVulnerability probability: {vuln_prob:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Prediction: {'VULNERABLE' if vuln_prob >= threshold else 'SAFE'}")


if __name__ == "__main__":
    test_dummy()
