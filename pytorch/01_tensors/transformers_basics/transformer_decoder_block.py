import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))

        cross_attn_output, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))

        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_output))
        return tgt

tgt = torch.randn(2, 5, 32)
memory = torch.randn(2, 7, 32)

block = TransformerDecoderBlock()
output = block(tgt, memory)

print("Target Shape:", tgt.shape)
print("Memory Shape:", memory.shape)
print("Output Shape:", output.shape)
