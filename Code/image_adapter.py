from torchvision import transforms
from PIL import Image

class ImageAdapter:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def adapter_input(self, image_path: str):
        img = Image.open(image_path).convert("L")
        return self.transform(img).unsqueeze(0)  # shape: (1, 1, 28, 28)
