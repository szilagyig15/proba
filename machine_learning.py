import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(x, coefficients, seed=42, noise_std=1):
    np.random.seed(seed)
    y = np.polyval(coefficients[::-1], x) + np.random.normal(0, noise_std, len(x))
    return x, y