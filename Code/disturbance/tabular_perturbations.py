# perturbations/tabular_perturbations.py

import numpy as np

def add_gaussian_noise(x, std=0.05):
    noise = np.random.normal(0, std, size=x.shape)
    return x + noise

def slight_scaling(x, factor=0.02):
    return x * (1 + np.random.uniform(-factor, factor))

def rounding(x, decimals=2):
    return np.round(x, decimals=decimals)
