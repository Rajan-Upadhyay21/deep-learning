import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

torch.save(model.state_dict(), "model_state_dict.pth")
print("Saved state_dict.")

loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load("model_state_dict.pth"))
loaded_model.eval()
print("Loaded state_dict into new model.")

torch.save(model, "full_model.pth")
print("Saved full model.")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "epoch": 1
}
torch.save(checkpoint, "checkpoint.pth")
print("Saved checkpoint.")

example_input = torch.randn(1, 4)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced_model.pt")
print("Saved TorchScript model.")
