# Full MAHA Transformer Architecture
import torch
import torch.nn as nn
from src.models.maha_block import MAHABlock

class MAHATransformer(nn.Module):
    """
    Full MAHA Transformer Architecture.
    
    It stacks L MAHA Blocks to form a powerful encoder for NLP tasks.
    Compatible with tasks like Text Classification (GLUE) or Masked Language Modeling.
    
    Ref: Section 5.1 Experimental Setup 
         - 12 Layers
         - 768 Hidden Dimension
         - 12 Attention Heads
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        num_scales: int = 4,
        aggregation_strategy: str = 'convex',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token & Position Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Stack of MAHA Blocks
        self.layers = nn.ModuleList([
            MAHABlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                num_scales=num_scales,
                aggregation_strategy=aggregation_strategy
            )
            for _ in range(num_layers)
        ])
        
        # Final Norm (Pre-classifier)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Example Classifier Head (Optional, for GLUE tasks)
        self.classifier = nn.Linear(d_model, vocab_size) # Or num_classes

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input tokens (Batch, Seq_Len)
            mask: Attention mask
        """
        B, N = x.size()
        
        # Embeddings
        positions = torch.arange(0, N, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        total_aux_loss = 0.0
        
        # Pass through MAHA Layers
        for layer in self.layers:
            x, layer_loss = layer(x, mask)
            total_aux_loss += layer_loss
            
        x = self.final_norm(x)
        
        # Return features and aggregated sparsity loss for optimization
        return x, total_aux_loss