import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# CUDA Hatalarını Yakalamak için Debug Ortamı
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.models.transformer import MAHATransformer

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Deneyler şu cihazda çalıştırılıyor: {device}")

def benchmark_efficiency():
    """
    DENEY 1: Hesaplama Verimliliği (Computational Efficiency)
    """
    print("\n🧪 DENEY 1: Verimlilik Testi (Standard Attention vs MAHA)...")
    
    # Not: Bellek hatası alırsanız 2048'i çıkarın
    seq_lengths = [128, 256, 512, 1024, 2048] 
    d_model = 256
    num_heads = 4
    batch_size = 4  
    
    results = []

    for seq_len in seq_lengths:
        # --- 1. Standard Transformer (Baseline) ---
        try:
            baseline = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads).to(device)
            dummy_input = torch.randn(seq_len, batch_size, d_model).to(device) 
            
            # Warmup
            _ = baseline(dummy_input, dummy_input, dummy_input)
            
            # Timing
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            with torch.no_grad():
                for _ in range(20): # Tekrar sayısı düşürüldü (Hız için)
                    _ = baseline(dummy_input, dummy_input, dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            
            baseline_time = (end - start) / 20
            baseline_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
            
            results.append({
                "Model": "Standard MHA",
                "Seq_Len": seq_len,
                "Time (ms)": baseline_time * 1000,
                "Memory (MB)": baseline_mem
            })
            
            del baseline, dummy_input
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ⚠️ Hata (Baseline - {seq_len}): {e}")

        # --- 2. MAHA Transformer ---
        try:
            # KRİTİK DÜZELTME: max_len parametresi artırıldı
            maha_model = MAHATransformer(
                vocab_size=100, 
                d_model=d_model, 
                num_heads=num_heads, 
                num_scales=4,
                aggregation_strategy='convex',
                max_len=5000  # <--- BURASI DÜZELTİLDİ (seq_len 2048'i kapsıyor)
            ).to(device)
            
            dummy_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
            
            # Warmup
            _ = maha_model(dummy_ids)
            
            # Timing
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = maha_model(dummy_ids)
            torch.cuda.synchronize()
            end = time.time()
            
            maha_time = (end - start) / 20
            maha_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            results.append({
                "Model": "MAHA (Ours)",
                "Seq_Len": seq_len,
                "Time (ms)": maha_time * 1000,
                "Memory (MB)": maha_mem
            })
            
            del maha_model, dummy_ids
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ⚠️ Hata (MAHA - {seq_len}): {e}")

        print(f"   -> Tamamlandı: Seq Len {seq_len}")

    return pd.DataFrame(results)

def ablation_study():
    """
    DENEY 2: Ablation Study (Convex vs Nash)
    """
    print("\n🧪 DENEY 2: Ablation Study (Convex vs Nash)...")
    
    strategies = ['convex', 'nash']
    loss_history = {s: [] for s in strategies}
    
    vocab_size = 500
    seq_len = 64
    d_model = 64
    epochs = 5
    
    train_data = torch.randint(0, vocab_size, (100, seq_len)).to(device)
    targets = torch.randint(0, vocab_size, (100, seq_len)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    for strategy in strategies:
        print(f"   -> Strateji Eğitiliyor: {strategy.upper()}")
        try:
            model = MAHATransformer(
                vocab_size=vocab_size, 
                d_model=d_model, 
                num_heads=4, 
                num_scales=3,
                aggregation_strategy=strategy,
                num_layers=1,
                max_len=512 # Yeterli
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            
            for epoch in range(epochs):
                batch_loss = 0
                for i in range(0, 100, 10):
                    bx = train_data[i:i+10]
                    by = targets[i:i+10]
                    
                    optimizer.zero_grad()
                    out, aux_loss = model(bx)
                    
                    # Projeksiyon (Linear classifier yoksa manuel yap)
                    # MAHATransformer içinde classifier tanımlı varsayıyoruz
                    if hasattr(model, 'classifier'):
                         logits = model.classifier(out)
                    else:
                         # Fallback
                         logits = nn.Linear(d_model, vocab_size).to(device)(out)

                    main_loss = criterion(logits.view(-1, vocab_size), by.view(-1))
                    
                    # Nash bazen aux_loss'u tensor(0.) döndürür, shape hatası olmasın
                    total_loss = main_loss + 0.1 * aux_loss
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    batch_loss += total_loss.item()
                
                loss_history[strategy].append(batch_loss / 10)
                
            del model
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"   ⚠️ Hata ({strategy}): {e}")
            
    return loss_history

def plot_results(df_eff, loss_hist):
    """Grafikleri Çizer"""
    sns.set_style("whitegrid")
    
    # 1. Time Complexity
    if df_eff is not None and not df_eff.empty:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.lineplot(data=df_eff, x="Seq_Len", y="Time (ms)", hue="Model", marker="o", linewidth=2.5)
        plt.title("Computational Time vs Sequence Length")
        plt.ylabel("Inference Time (ms)")
        plt.xlabel("Sequence Length (n)")
        
        # 2. Memory Usage
        plt.subplot(1, 2, 2)
        sns.barplot(data=df_eff, x="Seq_Len", y="Memory (MB)", hue="Model")
        plt.title("Peak Memory Usage vs Sequence Length")
        plt.ylabel("Memory (MB)")
        
        plt.tight_layout()
        plt.savefig("experiment_efficiency.png", dpi=300)
        print("\n✅ Verimlilik Grafiği Kaydedildi: experiment_efficiency.png")
    else:
        print("⚠️ Verimlilik verisi boş, grafik çizilemedi.")

    # 3. Ablation Loss
    if loss_hist:
        plt.figure(figsize=(8, 5))
        for strat, losses in loss_hist.items():
            if losses: # Liste boş değilse
                plt.plot(losses, label=f"Strategy: {strat}", marker='s')
        plt.title("Training Convergence: Convex vs Nash")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("experiment_ablation.png", dpi=300)
        plt.savefig("experiment_ablation.pdf")
        print("✅ Ablasyon Grafiği Kaydedildi: experiment_ablation.png")

if __name__ == "__main__":
    # Önceki hatalardan kalan context'i temizlemek gerekebilir.
    # Terminali kapatıp açmanız en iyisidir, ama kod içinde try-except var.
    
    df_results = benchmark_efficiency()
    if df_results is not None:
        print("\n--- Verimlilik Sonuçları ---")
        print(df_results)

    loss_history = ablation_study()
    
    plot_results(df_results, loss_history)