# Hybrid Dilated-Convolutional Transformer Block
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Önceki adımlarda yazdığımız modülleri import ediyoruz
from src.layers.decomposition import HierarchicalDecomposition
from src.layers.attention import MultiscaleAttention
from src.layers.aggregation import OptimizationDrivenAggregation

class MAHABlock(nn.Module):
    """
    Implements the Hybrid Dilated-Convolutional Transformer Block (Section 4.4).
    
    This block replaces the standard Self-Attention layer with the MAHA pipeline:
    1. Dilated Convolution (Local Context) [cite: 142]
    2. Hierarchical Decomposition [cite: 102]
    3. Multiscale Attention with Shared Values [cite: 114]
    4. Optimization-Driven Aggregation (Convex/Nash) [cite: 128]
    5. Feed-Forward Network (Standard)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        num_scales: int = 4,
        compression_ratio: int = 2,
        aggregation_strategy: str = 'convex'
    ):
        super().__init__()
        
        # 1. Pre-processing: Dilated Convolution to capture local context features
        # Eq (11): C_l = ReLU(DilatedConv(X...))
        # We apply a mild dilation here to enrich features before decomposition
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. MAHA Components
        self.decomposition = HierarchicalDecomposition(
            d_model=d_model,
            num_scales=num_scales,
            compression_ratio=compression_ratio,
            mode='conv'
        )
        
        self.attention = MultiscaleAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_scales=num_scales,
            decomposition_module=self.decomposition
        )
        
        self.aggregation = OptimizationDrivenAggregation(
            num_scales=num_scales,
            d_model=d_model,
            strategy=aggregation_strategy
        )
        
        # 3. Standard Transformer Components (Norm & FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (Batch, Seq_Len, d_model)
            mask: Attention mask
            
        Returns:
            output: Tensor (Batch, Seq_Len, d_model)
            aux_loss: Sparsity/Optimization loss from aggregation
        """
        # Residual Connection 1 (MAHA Branch)
        residual = x
        
        # A. Dilated Convolution (Requires Transpose for Conv1d)
        # x: (B, N, D) -> (B, D, N)
        x_conv = x.transpose(1, 2)
        x_conv = self.dilated_conv(x_conv)
        x_conv = x_conv.transpose(1, 2) # Back to (B, N, D)
        
        # B. Hierarchical Decomposition
        # x -> [X_0, X_1, ..., X_L]
        scales = self.decomposition(x_conv)
        
        # C. Multiscale Attention
        # [X_0...X_L] -> [O_0...O_L]
        attn_outputs = self.attention(scales, mask)
        
        # D. Aggregation (Convex or Nash)
        # [O_0...O_L] -> O*
        maha_out, aux_loss = self.aggregation(attn_outputs)
        
        # Apply projection and dropout
        maha_out = self.dropout(maha_out)
        
        # Add & Norm
        x = self.norm1(residual + maha_out)
        
        # Residual Connection 2 (FFN Branch)
        residual = x
        ffn_out = self.ffn(x)
        x = self.norm2(residual + ffn_out)
        
        return x, aux_loss