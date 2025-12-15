import unittest
import torch
import sys
import os

# Proje kök dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers.decomposition import HierarchicalDecomposition
from src.layers.attention import MultiscaleAttention
from src.layers.aggregation import OptimizationDrivenAggregation
from src.models.maha_block import MAHABlock
from src.models.transformer import MAHATransformer

class TestMAHAComponents(unittest.TestCase):
    
    def setUp(self):
        """Test ortamı için ortak parametreler"""
        self.batch_size = 2
        self.seq_len = 128  
        self.d_model = 64  # Küçük bir model boyutu
        self.num_heads = 4 # 64'e bölünebilir (64 % 4 == 0)
        self.num_scales = 3
        self.r = 2 
        
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_hierarchical_decomposition_shapes(self):
        """Test: Hiyerarşik ayrıştırma doğru boyutlarda tensör üretiyor mu?"""
        decomp = HierarchicalDecomposition(self.d_model, self.num_scales, self.r, mode='conv')
        outputs = decomp(self.dummy_input)
        
        self.assertEqual(len(outputs), self.num_scales)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.seq_len, self.d_model))
        expected_len_1 = self.seq_len // self.r
        self.assertEqual(outputs[1].shape[1], expected_len_1)
        print(f"✅ Decomposition Passed. Shapes: {[t.shape for t in outputs]}")

    def test_multiscale_attention_flow(self):
        """Test: Attention katmanı Shared Value ile çalışıyor mu?"""
        decomp = HierarchicalDecomposition(self.d_model, self.num_scales, self.r)
        attn = MultiscaleAttention(self.d_model, self.num_heads, self.num_scales, decomp)
        
        scales = decomp(self.dummy_input)
        attn_outputs = attn(scales)
        
        for i, out_tensor in enumerate(attn_outputs):
            self.assertEqual(out_tensor.shape, scales[i].shape)
        print("✅ Attention Passed.")

    def test_aggregation_convex(self):
        """Test: Convex Optimization birleştirme"""
        aggregator = OptimizationDrivenAggregation(self.num_scales, self.d_model, strategy='convex')
        inputs = [
            torch.randn(self.batch_size, self.seq_len, self.d_model),
            torch.randn(self.batch_size, self.seq_len // 2, self.d_model),
            torch.randn(self.batch_size, self.seq_len // 4, self.d_model)
        ]
        output, loss = aggregator(inputs)
        self.assertEqual(output.shape, inputs[0].shape)
        self.assertTrue(torch.is_tensor(loss))
        print("✅ Aggregation (Convex) Passed.")

    def test_aggregation_nash(self):
        """Test: Nash Equilibrium stratejisi"""
        aggregator = OptimizationDrivenAggregation(self.num_scales, self.d_model, strategy='nash', nash_iterations=2)
        inputs = [
            torch.randn(self.batch_size, self.seq_len, self.d_model),
            torch.randn(self.batch_size, self.seq_len // 2, self.d_model),
            torch.randn(self.batch_size, self.seq_len // 4, self.d_model)
        ]
        output, _ = aggregator(inputs)
        self.assertEqual(output.shape, inputs[0].shape)
        print("✅ Aggregation (Nash) Passed.")

    def test_full_maha_block(self):
        """Test: MAHABlock (End-to-End)"""
        block = MAHABlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_model * 4,
            num_scales=self.num_scales
        )
        output, aux_loss = block(self.dummy_input)
        self.assertEqual(output.shape, self.dummy_input.shape)
        print("✅ MAHABlock Passed.")
        
    def test_transformer_integration(self):
        """Test: Tam Transformer modeli"""
        vocab_size = 100
        # DÜZELTME BURADA: num_heads parametresini açıkça veriyoruz
        model = MAHATransformer(
            vocab_size=vocab_size, 
            d_model=self.d_model, 
            num_heads=self.num_heads,  # <--- EKLENDİ (64 % 4 == 0)
            num_layers=2
        )
        
        input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        output, total_loss = model(input_ids)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        print("✅ MAHATransformer Integration Passed.")

if __name__ == '__main__':
    unittest.main()