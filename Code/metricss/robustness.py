import numpy as np
import torch
import torch.nn.functional as F
from metricss.similarity import cosine_similarity
from disturbance.text_perturbations import text_typo_noise, word_dropout
from disturbance.image_perturbations import add_noise
from disturbance.tabular_perturbations import add_gaussian_noise

def robustness_test_tabular(model, x, output_adapter, n=5):
    scores = []

    y_orig = output_adapter.adapt_output(model.predict_proba(x)[0])

    for _ in range(n):
        x_pert = add_gaussian_noise(x)
        y_pert = output_adapter.adapt_output(model.predict_proba(x_pert)[0])
        score = cosine_similarity(y_orig, y_pert)
        scores.append(score)

    return float(np.mean(scores)), scores

def robustness_test_image(model, img_tensor, output_adapter, n=5):
    scores = []

    model.eval()

    # Original prediction (logits -> softmax probs)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        y_orig = output_adapter.adapt_output(probs)

    # Perturb image n times
    for _ in range(n):
        pert = add_noise(img_tensor)

        with torch.no_grad():
            logits_pert = model(pert)
            probs_pert = F.softmax(logits_pert, dim=1)
            y_pert = output_adapter.adapt_output(probs_pert)

        score = cosine_similarity(y_orig, y_pert)
        scores.append(score)

    return float(np.mean(scores)), scores


def robustness_test_text(model, text, output_adapter, n=5):

    scores = []

    # Original prediction
    y_orig = output_adapter.adapt_output(model.predict_proba([text])[0])

    for _ in range(n):
        # Apply noise perturbations (typos + word dropout)
        t = text_typo_noise(text)
        t = word_dropout(t)

        y_pert = output_adapter.adapt_output(model.predict_proba([t])[0])
        score = cosine_similarity(y_orig, y_pert)
        scores.append(score)

    return float(np.mean(scores)), scores


