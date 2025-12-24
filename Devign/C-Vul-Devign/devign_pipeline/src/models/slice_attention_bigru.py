"""BiGRU with Slice Attention for Vulnerability Detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class TokenAttention(nn.Module):
    """Attention mechanism for pooling token-level representations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] with 1 for valid tokens, 0 for padding
        Returns:
            pooled: [batch, hidden_dim]
        """
        scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            # Check if entire sequence is masked (padding slice)
            all_masked = (mask.sum(dim=-1) == 0)  # [batch]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            all_masked = torch.zeros(scores.size(0), dtype=torch.bool, device=scores.device)
        
        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        
        # Handle all-masked case: use uniform weights instead of NaN
        if all_masked.any():
            uniform = torch.ones_like(weights) / weights.size(-1)
            weights = torch.where(all_masked.unsqueeze(-1), uniform, weights)
        
        weights = weights.unsqueeze(-1)  # [batch, seq_len, 1]
        pooled = (hidden_states * weights).sum(dim=1)  # [batch, hidden_dim]
        return pooled


class SliceAttention(nn.Module):
    """Attention mechanism for pooling slice-level representations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, slice_repr: torch.Tensor, slice_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            slice_repr: [batch, num_slices, hidden_dim]
            slice_mask: [batch, num_slices] with 1 for valid slices, 0 for padding
        Returns:
            pooled: [batch, hidden_dim]
        """
        scores = self.attention(slice_repr).squeeze(-1)  # [batch, num_slices]
        
        if slice_mask is not None:
            # Check if all slices are masked (should not happen but handle gracefully)
            all_masked = (slice_mask.sum(dim=-1) == 0)  # [batch]
            scores = scores.masked_fill(slice_mask == 0, float('-inf'))
        else:
            all_masked = torch.zeros(scores.size(0), dtype=torch.bool, device=scores.device)
        
        weights = F.softmax(scores, dim=-1)  # [batch, num_slices]
        
        # Handle all-masked case: use uniform weights
        if all_masked.any():
            uniform = torch.ones_like(weights) / weights.size(-1)
            weights = torch.where(all_masked.unsqueeze(-1), uniform, weights)
        
        weights = weights.unsqueeze(-1)  # [batch, num_slices, 1]
        pooled = (slice_repr * weights).sum(dim=1)  # [batch, hidden_dim]
        return pooled


class FeatureMLP(nn.Module):
    """MLP branch for vulnerability features."""
    
    def __init__(self, feat_dim: int, output_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, feat_dim]
        Returns:
            output: [batch, output_dim]
        """
        return self.mlp(features)


class SliceAttBiGRU(nn.Module):
    """
    Hierarchical BiGRU with:
    1. Token-level attention within each slice
    2. Slice-level attention across slices (masked by slice_mask)
    3. Vulnerability features MLP branch
    4. Fusion and binary classification
    """
    
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        feat_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        embed_dropout: float = 0.3,
        gru_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        feat_dropout: float = 0.5,
        feat_output_dim: int = 64,
        classifier_hidden: int = 256,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional_dim = hidden_dim * 2  # 256
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=self.PAD_IDX
        )
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        self.bigru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if num_layers > 1 else 0
        )
        
        self.token_attention = TokenAttention(self.bidirectional_dim)
        self.token_layer_norm = nn.LayerNorm(self.bidirectional_dim)
        self.slice_attention = SliceAttention(self.bidirectional_dim)
        self.slice_layer_norm = nn.LayerNorm(self.bidirectional_dim)
        self.feature_mlp = FeatureMLP(feat_dim, feat_output_dim, feat_dropout)
        
        fusion_dim = self.bidirectional_dim + feat_output_dim  # 256 + 64 = 320
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.bigru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slice_mask: torch.Tensor,
        vuln_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, num_slices, seq_len] (Long)
            attention_mask: [batch, num_slices, seq_len] (0/1 for token padding)
            slice_mask: [batch, num_slices] (0/1 for slice padding)
            vuln_features: [batch, feat_dim] (float)
        Returns:
            logits: [batch, 1]
        """
        batch_size, num_slices, seq_len = input_ids.shape
        
        # 1. Flatten slices: [B, 4, 256] -> [B*4, 256]
        flat_input_ids = input_ids.view(batch_size * num_slices, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_slices, seq_len)
        
        # 2. Embedding + Dropout + BiGRU: [B*4, 256] -> [B*4, 256, 256]
        embeddings = self.embedding(flat_input_ids)  # [B*4, 256, 128]
        embeddings = self.embed_dropout(embeddings)
        gru_output, _ = self.bigru(embeddings)  # [B*4, 256, 256]
        
        # 3. Token attention pool + LayerNorm: [B*4, 256, 256] -> [B*4, 256]
        slice_repr = self.token_attention(gru_output, flat_attention_mask)  # [B*4, 256]
        slice_repr = self.token_layer_norm(slice_repr)
        
        # 4. Reshape: [B*4, 256] -> [B, 4, 256]
        slice_repr = slice_repr.view(batch_size, num_slices, self.bidirectional_dim)
        
        # 5. Slice attention pool + LayerNorm (with slice_mask): [B, 4, 256] -> [B, 256]
        code_repr = self.slice_attention(slice_repr, slice_mask)  # [B, 256]
        code_repr = self.slice_layer_norm(code_repr)
        
        # 6. Feature MLP: [B, F] -> [B, 64]
        feat_repr = self.feature_mlp(vuln_features)  # [B, 64]
        
        # 7. Concat + classify: [B, 320] -> [B, 1]
        fused = torch.cat([code_repr, feat_repr], dim=-1)  # [B, 320]
        logits = self.classifier(fused)  # [B, 1]
        
        return logits


def create_model(config: Dict[str, Any]) -> SliceAttBiGRU:
    """
    Create model from config dictionary.
    
    Args:
        config: Dictionary with model hyperparameters:
            - vocab_size (required)
            - emb_dim (default: 128)
            - hidden_dim (default: 128)
            - feat_dim (default: 64)
            - num_layers (default: 1)
            - dropout (default: 0.3)
            - embed_dropout (default: 0.3)
            - gru_dropout (default: 0.3)
            - classifier_dropout (default: 0.5)
            - feat_dropout (default: 0.5)
            - feat_output_dim (default: 64)
            - classifier_hidden (default: 256)
    
    Returns:
        SliceAttBiGRU model instance
    """
    return SliceAttBiGRU(
        vocab_size=config["vocab_size"],
        emb_dim=config.get("emb_dim", 128),
        hidden_dim=config.get("hidden_dim", 128),
        feat_dim=config.get("feat_dim", 64),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.3),
        embed_dropout=config.get("embed_dropout", 0.3),
        gru_dropout=config.get("gru_dropout", 0.3),
        classifier_dropout=config.get("classifier_dropout", 0.5),
        feat_dropout=config.get("feat_dropout", 0.5),
        feat_output_dim=config.get("feat_output_dim", 64),
        classifier_hidden=config.get("classifier_hidden", 256),
    )


if __name__ == "__main__":
    # Quick test
    batch_size, num_slices, seq_len = 2, 4, 256
    vocab_size, feat_dim = 5000, 32
    
    model = create_model({"vocab_size": vocab_size, "feat_dim": feat_dim})
    
    input_ids = torch.randint(0, vocab_size, (batch_size, num_slices, seq_len))
    attention_mask = torch.ones(batch_size, num_slices, seq_len)
    slice_mask = torch.ones(batch_size, num_slices)
    vuln_features = torch.randn(batch_size, feat_dim)
    
    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
    print(f"Input shapes: input_ids={input_ids.shape}, vuln_features={vuln_features.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
