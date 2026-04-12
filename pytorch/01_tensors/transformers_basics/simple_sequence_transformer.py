import torch
import torch.nn as nn

class SimpleSequenceTransformer(nn.Module):
    def __init__(self, vocab_size=50, d_model=32, num_heads=4, num_layers=2, max_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return self.output_layer(x)

model = SimpleSequenceTransformer()
tokens = torch.randint(0, 50, (2, 8))
output = model(tokens)

print("Input Shape:", tokens.shape)
print("Output Shape:", output.shape)
