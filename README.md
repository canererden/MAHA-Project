# MAHA: Multiscale Aggregated Hierarchical Attention

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

Official PyTorch implementation of the paper:
**"Multiscale Aggregated Hierarchical Attention (MAHA): A Game-Theoretic and Optimization-Driven Approach to Efficient Contextual Modeling in Large Language Models"**

## 📄 Abstract

We propose **MAHA**, a novel attention mechanism that reformulates multi-head self-attention through hierarchical multiscale decomposition and mathematically rigorous aggregation (Convex Optimization & Nash Equilibrium). MAHA achieves **sub-quadratic complexity** and superior long-range dependency modeling compared to standard Transformers.

## 🚀 Key Features

- **Hierarchical Decomposition:** Learnable Strided Convolutions to create multiscale representations.
- **Shared Value Projection:** Parameter-efficient attention computation.
- **Optimization-Driven Aggregation:**
  - `convex`: L1-regularized weighted sum.
  - `nash`: Iterative best-response equilibrium.
- **Hybrid Architecture:** Dilated Convolutional Transformer Block.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/username/maha-project.git](https://github.com/username/maha-project.git)
cd maha-project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training

To train the MAHA Transformer on synthetic data (proof-of-concept):

```bash
python train.py
```

### 2. Unit Tests

To verify architectural integrity and tensor shapes:

```bash
python -m unittest tests/test_maha.py
```

### 3. Using MAHA in Your Code

```python
import torch
from src.models.transformer import MAHATransformer

# Initialize Model
model = MAHATransformer(
    vocab_size=30000,
    max_len=1024,
    d_model=768,
    num_heads=12,
    num_scales=4,
    aggregation_strategy='convex'
)

# Forward Pass
inputs = torch.randint(0, 30000, (1, 1024))
output, aux_loss = model(inputs)

print(f"Output Shape: {output.shape}")  # (1, 1024, 768)
```

## 📊 Directory Structure

```
maha-project/
├── src/
│   ├── layers/          # Core MAHA layers (Decomposition, Attention, Aggregation)
│   ├── models/          # MAHABlock and Transformer architecture
│   └── optimization/    # Solvers
├── tests/               # Unit tests
├── configs/             # Hyperparameter configurations
└── train.py             # Training loop
```

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@article{maha2025,
  title={Multiscale Aggregated Hierarchical Attention (MAHA)},
  author={Erden, Caner},
  journal={Submitted to IEEE TNNLS},
  year={2025}
}
```
