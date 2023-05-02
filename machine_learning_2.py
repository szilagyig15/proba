import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_polinomial_data(coefficients, fromX, toX, n_samples, noise, random_state=None, filepath=None):
    np.random.seed(random_state)
    X = np.random.uniform(fromX, toX, n_samples)
    y = np.polyval(coefficients[::-1], X) + noise + np.random.randn(n_samples)
    if filepath:
        df = pd.DataFrame({'x':X, 'y':y})
        df.to_csv(filepath, index=False, header=False)
    return X.reshape(-1, 1), y
# y=100+1*x+0.2x**2
coeffs=[100, 1, 0.2]
X,y =generate_polinomial_data(coeffs, fromX=-5, toX=7, n_samples=500, noise=1, random_state=42, filepath='data.csv')
plt.scatter(X, y, label='Data', alpha=0.5)
plt.show()




