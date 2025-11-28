import torchvision.transforms.functional as TF
import torch
import random

def add_noise(img, std=0.1):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0, 1)

def rotate(img, degrees=10):
    deg = random.uniform(-degrees, degrees)
    return TF.rotate(img, deg)

def change_brightness(img, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    return TF.adjust_brightness(img, factor)
