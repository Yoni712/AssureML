import torch.nn.functional as F
import torch
from metricss.similarity import cosine_similarity
from disturbance.text_perturbations import lowercase, word_dropout, text_typo_noise
from disturbance.image_perturbations import rotate, change_brightness
from disturbance.tabular_perturbations import slight_scaling, rounding

def consistency_test_tabular(model, x, output_adapter):
    # 1. Compute original output
    y_orig = output_adapter.adapt_output(model.predict_proba(x)[0])

    # 2. Compute consistency for each perturbation type
    results = {}

    v_scale = slight_scaling(x)
    y_scale = output_adapter.adapt_output(model.predict_proba(v_scale)[0])
    results["scaling"] = cosine_similarity(y_orig, y_scale)

    v_round = rounding(x)
    y_round = output_adapter.adapt_output(model.predict_proba(v_round)[0])
    results["rounding"] = cosine_similarity(y_orig, y_round)

    # 3. Calculate average consistency
    avg_consistency = sum(results.values()) / len(results)

    # 4. Return BOTH average and breakdown
    return avg_consistency, results

def consistency_test_image(model, img_tensor, output_adapter):
    model.eval()

    with torch.no_grad():
        logits_orig = model(img_tensor)
        probs_orig = F.softmax(logits_orig, dim=1)
        y_orig = output_adapter.adapt_output(probs_orig)

    results = {}

    # Rotation
    img_r = rotate(img_tensor)
    with torch.no_grad():
        logits_r = model(img_r)
        probs_r = F.softmax(logits_r, dim=1)
        y_r = output_adapter.adapt_output(probs_r)
    results["rotation"] = cosine_similarity(y_orig, y_r)

    # Brightness
    img_b = change_brightness(img_tensor)
    with torch.no_grad():
        logits_b = model(img_b)
        probs_b = F.softmax(logits_b, dim=1)
        y_b = output_adapter.adapt_output(probs_b)
    results["brightness"] = cosine_similarity(y_orig, y_b)

    avg = sum(results.values()) / len(results)

    return float(avg), results


def consistency_test_text(model, text, output_adapter):

    y_orig = output_adapter.adapt_output(model.predict_proba([text])[0])

    results = {}

    # Lowercase version (meaning preserved)
    t1 = lowercase(text)
    y1 = output_adapter.adapt_output(model.predict_proba([t1])[0])
    results["lowercase"] = cosine_similarity(y_orig, y1)

    # Very small typo noise (barely meaning-changing)
    t2 = text_typo_noise(text, prob=0.005)
    y2 = output_adapter.adapt_output(model.predict_proba([t2])[0])
    results["tiny_typo_noise"] = cosine_similarity(y_orig, y2)

    avg_score = sum(results.values()) / len(results)

    return float(avg_score), results

