import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal

class OptimizationDrivenAggregation(nn.Module):
    """
    Implements Optimization-Driven Aggregation (Section 4.3 of MAHA paper).
    
    Strategies:
    1. Convex Optimization (Eq 9): Learns weights w s.t. sum(w)=1, w>=0 with L1 sparsity.
    2. Nash Equilibrium (Eq 10): Iterative best-response dynamics to find equilibrium weights.
    
    Includes Nearest-Neighbor Upsampling (Eq 13) to align scales.
    """
    
    def __init__(
        self, 
        num_scales: int, 
        d_model: int,
        strategy: Literal['convex', 'nash'] = 'convex',
        nash_iterations: int = 3,
        lambda_sparsity: float = 0.1
    ):
        super().__init__()
        self.num_scales = num_scales
        self.strategy = strategy
        self.nash_iterations = nash_iterations
        self.lambda_sparsity = lambda_sparsity
        
        # Learnable weights for Convex strategy
        # Initialized to be uniform (1/L) after softmax
        self.convex_logits = nn.Parameter(torch.zeros(num_scales))
        
        # Linear projection for Nash utility estimation (optional but improves stability)
        self.utility_proj = nn.Linear(d_model, 1) if strategy == 'nash' else None

    def _upsample(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Nearest-Neighbor Upsampling (Eq 13).
        Input: (B, N_l, D) -> Output: (B, N_target, D)
        """
        # Permute for interpolate: (B, D, N)
        tensor_p = tensor.transpose(1, 2)
        
        # Apply Nearest Neighbor interpolation
        upsampled = F.interpolate(
            tensor_p, 
            size=target_len, 
            mode='nearest'
        )
        
        # Permute back: (B, N, D)
        return upsampled.transpose(1, 2)

    def solve_convex(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements Convex Optimization aggregation.
        Returns: (Aggregated_Tensor, Sparsity_Loss)
        """
        # Enforce constraints: sum(w)=1, w>=0 via Softmax
        weights = F.softmax(self.convex_logits, dim=0)
        
        # Calculate L1 sparsity loss (Eq 9 penalty term)
        # We want weights to be sparse (some close to 0)
        sparsity_loss = self.lambda_sparsity * torch.norm(weights, p=1)
        
        # Weighted Sum
        # Ensure base tensor is on correct device and shape
        final_output = torch.zeros_like(outputs[0])
        for i, out_tensor in enumerate(outputs):
            final_output += weights[i] * out_tensor
            
        return final_output, sparsity_loss

    def solve_nash(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements Nash Equilibrium aggregation via Best-Response Dynamics.
        """
        batch_size = outputs[0].shape[0]
        
        # Initialize weights uniformly: (B, L, 1)
        # Each sample in batch can have different equilibrium
        weights = torch.ones(batch_size, self.num_scales, 1, device=outputs[0].device) 
        weights = weights / self.num_scales
        
        # Iterative Best-Response (Unrolled Optimization)
        # We simulate 'nash_iterations' steps of players adjusting strategies
        stacked_outputs = torch.stack(outputs, dim=1) # (B, L, N, D)

        for _ in range(self.nash_iterations):
            # 1. Compute current Consensus (O*)
            consensus = (stacked_outputs * weights.unsqueeze(-1)).sum(dim=1) # (B, N, D)
            
            # 2. Compute Utility/Error for each scale
            # Error_l = || O_l - O* ||^2
            # We want to minimize reconstruction error
            diffs = stacked_outputs - consensus.unsqueeze(1) # (B, L, N, D)
            errors = torch.norm(diffs, p=2, dim=(2, 3)) # (B, L)
            
            # 3. Update weights (Softmax over negative error -> minimized error gets higher weight)
            # This is a differentiable approximation of argmin
            weights = F.softmax(-errors, dim=1).unsqueeze(-1) # (B, L, 1)

        # Final Aggregation using Equilibrium Weights
        final_output = (stacked_outputs * weights.unsqueeze(-1)).sum(dim=1)
        
        return final_output, torch.tensor(0.0, device=final_output.device)

    def forward(self, scale_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scale_outputs: List of tensors [O_0, O_1, ..., O_L] with different lengths.
            
        Returns:
            (Aggregated_Tensor, Aux_Loss)
        """
        target_len = scale_outputs[0].size(1)
        
        # 1. Upsample all scales to target length (Scale 0 length)
        upsampled_outputs = [scale_outputs[0]]
        for i in range(1, self.num_scales):
            upsampled_outputs.append(
                self._upsample(scale_outputs[i], target_len)
            )
            
        # 2. Aggregate based on strategy
        if self.strategy == 'convex':
            return self.solve_convex(upsampled_outputs)
        elif self.strategy == 'nash':
            return self.solve_nash(upsampled_outputs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")