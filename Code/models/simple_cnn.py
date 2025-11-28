import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(8*222*222, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1).detach().cpu().numpy()
