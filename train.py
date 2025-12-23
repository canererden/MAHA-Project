# Main Training Loop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from src.models.transformer import MAHATransformer

def train():
    # --- 1. Hyperparameters (Section 5.1 Setup) ---
    VOCAB_SIZE = 1000
    SEQ_LEN = 128
    BATCH_SIZE = 16
    D_MODEL = 256        # Reduced for demo speed
    NUM_HEADS = 4
    NUM_LAYERS = 2
    NUM_SCALES = 4
    EPOCHS = 5
    LR = 5e-5
    LAMBDA_REG = 0.1     # Sparsity regularization coefficient (lambda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # --- 2. Synthetic Data ---
    # Random integers simulating token IDs
    train_data = torch.randint(0, VOCAB_SIZE, (1000, SEQ_LEN))
    # Random targets (e.g., for Masked LM or Classification)
    train_labels = torch.randint(0, VOCAB_SIZE, (1000, SEQ_LEN))
    
    dataset = TensorDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. Model Initialization ---
    model = MAHATransformer(
        vocab_size=VOCAB_SIZE,
        max_len=SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_scales=NUM_SCALES,
        aggregation_strategy='convex' # Try 'nash' as well
    ).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"✅ Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. Training Loop ---
    model.train()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        total_task_loss = 0
        total_aux_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # output: (B, Seq, D_model), aux_loss: Scalar (sum of L1 norms)
            outputs, aux_loss = model(inputs)
            
            # Compute Task Loss (Flatten for CrossEntropy)
            # outputs: (B*Seq, Vocab), targets: (B*Seq)
            # Note: We need a projection to vocab size here if not in model
            # For this demo, we use the classifier head inside MAHATransformer if it existed,
            # or just project simply here:
            logits = model.classifier(outputs) # (B, Seq, Vocab)
            
            task_loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
            # Combine Losses (Eq 9 in Paper)
            loss = task_loss + (LAMBDA_REG * aux_loss)
            
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_aux_loss += aux_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} (Task: {task_loss.item():.4f} + Aux: {aux_loss.item():.4f})")
        
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"🏁 End of Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    train()