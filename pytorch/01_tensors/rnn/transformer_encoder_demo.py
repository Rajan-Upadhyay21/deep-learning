import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
d_model = 16
nhead = 4

x = torch.randn(batch_size, seq_len, d_model)

encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

output = transformer_encoder(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
