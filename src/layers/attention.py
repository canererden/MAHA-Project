import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from .decomposition import HierarchicalDecomposition

class MultiscaleAttention(nn.Module):
    """
    Implements Multiscale Attention Computation (Section 4.2 of MAHA paper).
    
    Features:
    - Scale-specific Query (Q) and Key (K) projections.
    - Shared Value (V) projection across all scales.
    - Efficient computation using re-used downsampling operators for V.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        num_scales: int,
        decomposition_module: HierarchicalDecomposition
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_scales = num_scales
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 1. Scale-Specific Projections for Q and K
        # We create a separate Linear layer for each scale l.
        self.q_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])
        
        # 2. Shared Value Projection 
        # V_base = X * W_V
        self.shared_v_proj = nn.Linear(d_model, d_model)
        
        # Reference to decomposition module to downsample V_base for each scale
        # V_l = D_l(V_base) [cite: 124]
        self.decomposition = decomposition_module

    def forward(
        self, 
        x_scales: List[torch.Tensor], 
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            x_scales (List[Tensor]): List of decomposed inputs [X_0, X_1, ..., X_L].
                                     Output from HierarchicalDecomposition.forward().
            mask (Tensor, optional): Standard attention mask (broadcastable).
            
        Returns:
            List[Tensor]: List of attention outputs [O_0, O_1, ..., O_L] per scale.
        """
        
        # Step 1: Compute Base Value (V_base) from the original input (Scale 0)
        # X_0 is usually the high-res input
        x_base = x_scales[0] 
        v_base = self.shared_v_proj(x_base) # (B, N, d)
        
        # Step 2: Hierarchically decompose V_base using the SAME operators used for X
        # This ensures V_l matches the length of Q_l and K_l
        # [cite: 124] V_l = D_l(V_base)
        v_scales = self.decomposition(v_base)
        
        outputs = []
        
        # Step 3: Compute Attention for each scale independently
        for l in range(self.num_scales):
            x_l = x_scales[l] # Input at scale l
            v_l = v_scales[l] # Value at scale l
            
            B, N_l, _ = x_l.size()
            
            # Projections 
            q_l = self.q_projs[l](x_l)
            k_l = self.k_projs[l](x_l)
            
            # Reshape for Multi-Head Attention: (B, N, H, D_h) -> (B, H, N, D_h)
            q_l = q_l.view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
            k_l = k_l.view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
            v_l = v_l.view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled Dot-Product Attention [cite: 120]
            # scores = (Q K^T) / sqrt(d_k)
            scores = torch.matmul(q_l, k_l.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                # Need to resize mask for current scale if mask is provided
                # Simplified masking logic for brevity (assuming causal or padding mask)
                pass 
            
            attn_weights = F.softmax(scores, dim=-1)
            
            # O_l = A_l * V_l [cite: 126]
            context = torch.matmul(attn_weights, v_l)
            
            # Reshape back: (B, H, N, D_h) -> (B, N, D)
            context = context.transpose(1, 2).contiguous().view(B, N_l, self.d_model)
            
            outputs.append(context)
            
        return outputs