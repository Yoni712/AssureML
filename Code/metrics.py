import numpy as np
np.random.seed(42)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_noise(x, scale=0.1):
    return x + np.random.normal(0, scale, x.shape)

def robustness_test(model, x, output_adapter):
    x_noisy = add_noise(x)
    y1 = model.predict_proba([x])[0]
    y2 = model.predict_proba([x_noisy])[0]
    y1 = output_adapter.adapt_output(y1)
    y2 = output_adapter.adapt_output(y2)
    return cosine_similarity(y1, y2)

def generate_equivalent_samples(x):
    samples = []

    # 1. Scaling by a constant
    samples.append(x * 1.01)
    samples.append(x * 0.99)

    # 2. Rounding (keeps approx meaning)
    samples.append(np.round(x, 1))

    # 3. Small correlated noise
    samples.append(x + np.full(x.shape, 0.05))
    samples.append(x - np.full(x.shape, 0.05))

    # 4. Slight permutations of features (if allowed)
    samples.append(np.roll(x, 1))
    samples.append(np.roll(x, -1))

    return samples


def consistency_test(model, x, output_adapter):
    equivalents = generate_equivalent_samples(x)
    y_original = output_adapter.adapt_output(model.predict_proba([x])[0])

    similarities = []
    for e in equivalents:
        y_eq = output_adapter.adapt_output(model.predict_proba([e])[0])
        sim = np.dot(y_original, y_eq) / (np.linalg.norm(y_original) * np.linalg.norm(y_eq))
        similarities.append(sim)

    return np.mean(similarities)

def prediction_variance(model, x, output_adapter, n_runs=20, noise_scale=0.001):
    """
    Measures prediction variance by running multiple predictions on slightly
    noise-perturbed versions of the same input.
    """
    outputs = []

    for _ in range(n_runs):
        # Add tiny Gaussian noise
        x_noisy = x + np.random.normal(0, noise_scale, x.shape)

        # Get probability prediction
        y_pred = model.predict_proba([x_noisy])[0]

        # Adapt output to vector format
        y_pred = output_adapter.adapt_output(y_pred)

        outputs.append(y_pred)

    outputs = np.array(outputs)

    # Variance across runs (per class), then average
    return np.mean(np.var(outputs, axis=0))
