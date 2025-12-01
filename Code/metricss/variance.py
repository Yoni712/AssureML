import numpy as np
import torch.nn.functional as F
import torch
from disturbance.text_perturbations import text_typo_noise
from disturbance.image_perturbations import add_noise
from disturbance.tabular_perturbations import add_gaussian_noise

def variance_test_tabular(model, x, output_adapter, n_runs=10):
    outputs = []

    # Collect prediction vectors across multiple micro-noise runs
    for _ in range(n_runs):
        x_noisy = add_gaussian_noise(x, std=0.02)
        y = output_adapter.adapt_output(model.predict_proba(x_noisy)[0])
        outputs.append(y)

    outputs = np.array(outputs)

    # variance for each run (scalar per run)
    variances = outputs.var(axis=1)  # axis=1 gives 1 variance per run

    # return mean variance and list of variances
    return float(variances.mean()), list(variances)


def variance_test_image(model, img_tensor, output_adapter, n_runs=10):
    outputs = []
    model.eval()

    for _ in range(n_runs):
        pert = add_noise(img_tensor, std=0.02)

        with torch.no_grad():
            logits = model(pert)
            probs = F.softmax(logits, dim=1)
            y = output_adapter.adapt_output(probs)

        outputs.append(y)

    outputs = np.array(outputs)
    variances = outputs.var(axis=1)  # variance per run

    return float(variances.mean()), list(variances)

def variance_test_text(model, text, output_adapter, n_runs=10):
    
    outputs = []

    for _ in range(n_runs):
        # Very small typo noise to simulate micro input jitter
        t = text_typo_noise(text, prob=0.003)
        y = output_adapter.adapt_output(model.predict_proba([t])[0])
        outputs.append(y)

    outputs = np.array(outputs)
    variances = outputs.var(axis=1)

    return float(variances.mean()), list(variances)

