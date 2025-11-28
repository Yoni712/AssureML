import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Simple CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

        # MNIST normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load MNIST training data
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        # Train model for 1 epoch (enough for your thesis)
        self.train_model(train_loader)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 12 * 12)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_model(self, loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train()
        for images, labels in loader:
            optimizer.zero_grad()
            output = self.forward(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
        print("Image model trained (1 epoch)")

    def predict_proba(self, img_tensor):
        self.eval()
        with torch.no_grad():
            logits = self.forward(img_tensor)
        return F.softmax(logits, dim=1).numpy()
