import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScaleGating(nn.Module):
    """
    Implements Cross-Scale Gating (Eq 12 in MAHA paper).
    
    Mechanics:
    G_l = sigmoid(W_g * [C_l; U(C_{l+1})]) * C_l + (1 - sigmoid(...)) * U(C_{l+1})
    
    It dynamically blends information from the current scale and the coarser scale below it.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        # Input dimension is 2 * d_model because we concatenate current and next scale
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        
    def _upsample(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        # Re-use nearest neighbor logic for consistency
        tensor_p = tensor.transpose(1, 2)
        upsampled = F.interpolate(tensor_p, size=target_len, mode='nearest')
        return upsampled.transpose(1, 2)

    def forward(self, current_scale: torch.Tensor, next_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_scale (Tensor): Feature map at scale l (B, N_l, D)
            next_scale (Tensor): Feature map at scale l+1 (Coarser) (B, N_{l+1}, D)
            
        Returns:
            Tensor: Gated feature map at scale l (B, N_l, D)
        """
        # 1. Upsample coarser scale to match current scale
        target_len = current_scale.size(1)
        next_scale_up = self._upsample(next_scale, target_len)
        
        # 2. Concatenate [C_l; U(C_{l+1})]
        combined = torch.cat([current_scale, next_scale_up], dim=-1)
        
        # 3. Compute Gate Coefficient (z)
        # z = Sigmoid(W_g * combined)
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # 4. Apply Gating
        # G_l = z * C_l + (1 - z) * U(C_{l+1})
        output = gate * current_scale + (1 - gate) * next_scale_up
        
        return output