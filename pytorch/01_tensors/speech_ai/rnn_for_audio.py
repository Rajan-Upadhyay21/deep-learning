import torch
import torch.nn as nn

class AudioRNN(nn.Module):
    def __init__(self, input_size=13, hidden_size=32, num_layers=1, num_classes=4):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)

x = torch.randn(16, 25, 13)
model = AudioRNN()
output = model(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
