import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
X, y =generate_polinomial_data(coeffs, fromX=-5, toX=7, n_samples=500, noise=1, random_state=42, filepath='data.csv')
plt.scatter(X, y, label='Data', alpha=0.5)
plt.show()

# Demonstrate train and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_train_and_test_set(X_train, X_test, y_train, y_test):
    plt.scatter(X_train, y_train, label="Train", alpha=0.5)
    plt.scatter(X_test, y_test, label="Test", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Train-Test Split")
    plt.legend()
    plt.show()


plot_train_and_test_set(X_train, X_test, y_train, y_test)


def create_polynomial_model(degree=1):
    name = 'Polynomial_' + str(degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    return name, model

def creat_train_and_test_evaluate_polynomial_model(X_train, X_test, y_train, y_test, degree=15):
    name, model = create_polynomial_model(degree)
    model.fit(X_train, y_train)
    coefficients_on_train_set = model.named_steps['linearregression'].coef_
    y_pred = model.predict(X_test)
    mse_on_test_set = mean_squared_error(y_test, y_pred)
    return name, model, mse_on_test_set, coefficients_on_train_set


def hyperparameter_search(X_train, y_train, X_test, y_test, from_degree=1, to_degree=15):
    degrees = range(from_degree, to_degree)
    best_degree, best_mse, best_model = None, float('inf'), None
    d_mse = {}
    for degree in degrees:
        name, model, mse_on_test_set, coefficients_on_train_set = creat_train_and_test_evaluate_polynomial_model(X_train, y_train, X_test, y_test, degree=degree)
        d_mse[degree] = mse_on_test_set
        print(f'for degree: {degree}, MSE: {mse_on_test_set}')
        if mse_on_test_set < best_mse:
            best_degree, best_mse, best_model = degree, mse_on_test_set, model
    print(f'Best degree: {best_degree}, Best MSE: {best_mse}')
    print_coeffs('Coefficients: ', best_model)
    return best_model


def print_coeffs(text, model):
    if 'linear_regression' in model.named_steps.keys():
        linreg = 'linear_regression'
    else:
        linreg = 'linearregression'
    coeffs = np.concatenate(([model.named_steps[linreg].intercept_], model.named_steps[linreg].coef_[1:]))
    coeffs_str = ' '.join(np.format_float_positional(coeff, precision=4) for coeff in coeffs)
    print(text + coeffs_str)


