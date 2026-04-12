import torch
import torch.nn as nn

batch_size = 4
seq_len = 5
input_size = 2
hidden_size = 16

x = torch.randn(batch_size, seq_len, input_size)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

model = SimpleLSTM(input_size, hidden_size)
out = model(x)

print("Input Shape:", x.shape)
print("Output Shape:", out.shape)
print(out)
