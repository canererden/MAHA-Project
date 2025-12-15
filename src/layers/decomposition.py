import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

class HierarchicalDecomposition(nn.Module):
    """
    Implements Hierarchical Multiscale Decomposition (Section 4.1 of MAHA paper).
    
    This layer decomposes the input sequence X into L hierarchical scales using
    learnable downsampling operators (Strided Convolution) or Adaptive Pooling.
    
    Paper Reference:
        Eq (5): X_l = D_l(X_{l-1})
        Eq (109): Strided Convolution logic
        Eq (112): Exponential decay schedule n_l = floor(n_{l-1} / r)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_scales: int = 4, 
        compression_ratio: int = 2, 
        mode: Literal['conv', 'pool'] = 'conv',
        kernel_size: int = 3
    ):
        """
        Args:
            d_model (int): The embedding dimension (d).
            num_scales (int): Number of hierarchical scales (L).
            compression_ratio (int): Downsampling factor (r).
            mode (str): 'conv' for learnable Strided Convolution, 'pool' for Adaptive Max Pooling.
            kernel_size (int): Kernel size for convolution (default: 3).
        """
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.compression_ratio = compression_ratio
        self.mode = mode
        
        # We need (L-1) downsampling operators since Scale 0 is the original input.
        # Using ModuleList to register parameters properly.
        self.downsamplers = nn.ModuleList()
        
        if self.mode == 'conv':
            for _ in range(num_scales - 1):
                # Eq (109): D_l(X) = Conv1D(X, W_l, s_l)
                # Note: Groups=1 implies full interaction between channels as per W in R^{k x d x d}
                self.downsamplers.append(
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=kernel_size,
                        stride=compression_ratio,
                        padding=kernel_size // 2  # To maintain consistent alignment
                    )
                )
        elif self.mode == 'pool':
            # Pooling doesn't require learnable parameters per scale, 
            # but we keep the structure consistent.
            pass
        else:
            raise ValueError(f"Unknown decomposition mode: {mode}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, d_model)
            
        Returns:
            List[torch.Tensor]: A list of length `num_scales`.
                                Scale 0: (B, N, d)
                                Scale 1: (B, N/r, d)
                                ...
        """
        # x shape: [Batch, Length, Dim] -> Transpose for Conv1d: [Batch, Dim, Length]
        current_x = x.transpose(1, 2)
        outputs = [x] # Scale 0 is the input itself [cite: 106]
        
        for i in range(self.num_scales - 1):
            if self.mode == 'conv':
                # Apply learnable strided convolution
                # current_x represents X_{l-1}
                next_x = self.downsamplers[i](current_x)
                
            elif self.mode == 'pool':
                # Eq (111): Adaptive Pooling
                # Calculate target length: n_l = floor(n_{l-1} / r)
                current_len = current_x.size(2)
                target_len = current_len // self.compression_ratio
                
                # Prevent collapse to 0 length for very deep hierarchies
                target_len = max(1, target_len)
                
                next_x = F.adaptive_max_pool1d(current_x, output_size=target_len)
            
            # Update for next iteration
            current_x = next_x
            
            # Transpose back to [Batch, Length, Dim] for attention processing
            # and append to outputs list
            outputs.append(current_x.transpose(1, 2))
            
        return outputs

    def get_output_shapes(self, input_len: int) -> List[int]:
        """Helper to compute expected sequence lengths for debugging."""
        shapes = [input_len]
        curr = input_len
        for _ in range(self.num_scales - 1):
            curr = curr // self.compression_ratio
            shapes.append(curr)
        return shapes