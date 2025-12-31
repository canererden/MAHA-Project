# Multiscale Aggregated Hierarchical Attention (MAHA)

<div align="center">

![License](https://github.com/canererden/MAHA-Project/releases)
![Python](https://github.com/canererden/MAHA-Project/releases%2B-green)
![PyTorch](https://github.com/canererden/MAHA-Project/releases%2B-orange)
[![arXiv](https://github.com/canererden/MAHA-Project/releases)](https://github.com/canererden/MAHA-Project/releases)

**A Game-Theoretic and Optimization-Driven Approach to Efficient Contextual Modeling in Large Language Models**

[**Read the Paper**](https://github.com/canererden/MAHA-Project/releases)

</div>

---

## Abstract

We propose **MAHA**, a novel attention mechanism that reformulates multi-head self-attention through **hierarchical multiscale decomposition** and mathematically rigorous aggregation (**Convex Optimization** & **Nash Equilibrium**).

Standard attention mechanisms suffer from quadratic complexity $O(N^2)$. MAHA addresses this by dynamically partitioning the sequence into hierarchical scales and aggregating them using optimization solvers. The result is a framework that achieves **sub-quadratic complexity** and superior long-range dependency modeling compared to standard Transformers, specifically optimized for high-throughput inference.

## Architecture

MAHA replaces the standard Multi-Head Attention layer with a hierarchical processing block.

```mermaid
graph TD;
    Input[Input Sequence X] --> Decomp[Hierarchical Decomposition];
    Decomp -->|Scale 0| Attn0[Attention S0];
    Decomp -->|Scale 1| Attn1[Attention S1];
    Decomp -->|Scale 2| Attn2[Attention S2];
    Attn0 --> Upsample[Upsampling];
    Attn1 --> Upsample;
    Attn2 --> Upsample;
    Upsample --> Agg{Optimization Aggregator};
    Agg -->|Convex / Nash| Output[Aggregated Context];
    
    style Agg fill:#f9f,stroke:#333,stroke-width:2px
    style Decomp fill:#bbf,stroke:#333,stroke-width:2px

```

## Key Features

**Hierarchical Decomposition:** Uses learnable Strided Convolutions to create multiscale representations (scales l=1..L), reducing effective sequence length geometrically.
**Shared Value Projection:** Decouples Query/Key projections while sharing Value projections across scales, significantly reducing parameter count.
** Optimization-Driven Aggregation:**
**`convex` strategy:** Solves a constrained L1-regularized optimization problem to weigh scales.
**`nash` strategy:** Simulates a non-cooperative game where scales compete to minimize reconstruction error (Best-Response Dynamics).


**Hybrid Design:** Integrates Dilated Convolutional blocks for local feature extraction prior to attention.

## Performance

MAHA demonstrates superior efficiency on long-sequence tasks (e.g., PG-19) compared to standard baselines.

| Model | Complexity | PG-19 (PPL) $\downarrow$ | Memory Usage $\downarrow$ |
| --- | --- | --- | --- |
| Standard Transformer | O(N^2) | 24.3 | 15.2 GB |
| Longformer | O(N) | 23.8 | 9.1 GB |
| **MAHA (Ours)** | **Sub-Quadratic** | **23.1** | **6.7 GB** |

## Installation
```bash
# Clone the repository
git clone [https://github.com/canererden/MAHA-Project/releases](https://github.com/canererden/MAHA-Project/releases)
cd MAHA-Project

# Install dependencies
pip install -r https://github.com/canererden/MAHA-Project/releases

```

*Note: For the Convex Optimization solver, `cvxpylayers` is required.*

## Usage

### Quick Start
You can use `MAHABlock` as a drop-in replacement for standard attention layers or use the full `MAHATransformer` model.

```python
import torch
from https://github.com/canererden/MAHA-Project/releases import MAHATransformer

# Initialize Model with Convex Aggregation
model = MAHATransformer(
    vocab_size=30000,
    max_len=4096,        # Long context support
    d_model=768,
    num_heads=12,
    num_scales=4,        # L=4 scales (e.g., 4096, 2048, 1024, 512)
    aggregation_strategy='convex' # or 'nash'
)

# Move to GPU
device = https://github.com/canererden/MAHA-Project/releases("cuda" if https://github.com/canererden/MAHA-Project/releases() else "cpu")
https://github.com/canererden/MAHA-Project/releases(device)

# Forward Pass
dummy_input = https://github.com/canererden/MAHA-Project/releases(0, 30000, (1, 4096)).to(device)
output, aux_loss = model(dummy_input)

print(f"Output Shape: {https://github.com/canererden/MAHA-Project/releases}")  # (1, 4096, 768)

```

### Running Experiments

To replicate the training runs from the paper:

```bash
# Train on synthetic data or configured dataset
python https://github.com/canererden/MAHA-Project/releases --config https://github.com/canererden/MAHA-Project/releases

# Run Unit Tests
python -m unittest discover tests/

```

## Directory Structure
```text
maha-project/
├── configs/             # Hyperparameter configurations (YAML)
├── src/
│   ├── layers/          # Core MAHA layers (Decomposition, Attention, Aggregation)
│   ├── models/          # MAHABlock and Transformer architecture
│   ├── optimization/    # Differentiable solvers (Convex & Game Theory)
│   └── utils/           # Metrics and helpers
├── tests/               # Unit tests for tensor shapes and gradients
├── https://github.com/canererden/MAHA-Project/releases             # Main training loop
└── https://github.com/canererden/MAHA-Project/releases     # Dependencies

```

# Citation
If you use this code or our results in your research, please cite our work using the persistent **Zenodo DOI**:

```bibtex
@article{erden2025maha,
  title={Multiscale Aggregated Hierarchical Attention (MAHA): A Game Theoretic and Optimization Driven Approach to Efficient Contextual Modeling in Large Language Models},
  author={Erden, Caner},
  journal={arXiv preprint arXiv:2512.14925},
  year={2025},
  url={https://github.com/canererden/MAHA-Project/releases}
}

```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/canererden/MAHA-Project/releases) file for details.