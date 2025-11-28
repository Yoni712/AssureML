import torch
from torchvision.models import resnet18, ResNet18_Weights

# SAFE, reliable method (no hub)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

torch.save(model, "resnet18.pt")
print("Saved ResNet18 as resnet18.pt")
