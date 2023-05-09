import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def generate_polynomial_data(coefficients, fromX, toX, n_samples, noise, random_state=None, filepath=None):
    np.random.seed(random_state)

    X = np.random.uniform(fromX, toX, n_samples)
    y = np.polyval(coefficients[::-1], X) + noise * np.random.randn(n_samples)

    if filepath:
        df = pd.DataFrame({'x': X, 'y': y})
        df.to_csv(filepath, index=False, header=False)

    return X.reshape(-1, 1), y


# y = 100 + 1*x + 0.2*x**2
# coeffs = [100, 1, 0.2]
# X, y = generate_polinomial_data(coeffs, fromX=-5, toX=7, n_samples=500, noise=1, random_state=42)
# plt.scatter(X, y, label='Data', alpha=0.5)
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_train_and_test_split(X_train, X_test, y_train, y_test):
    plt.scatter(X_train, y_train, label='Train', alpha=0.5)
    plt.scatter(X_test, y_test, label='Test', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Train-Test Split')
    plt.legend()
    plt.show()


# plot_train_and_test_split(X_train, X_test, y_train, y_test)


def create_polynomial_model(degree=1):
    name = 'Polynomial_' + str(degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    return name, model


def create_train_and_evaluate_polynomial_model(X_train, y_train, X_test, y_test, degree=15):
    name, model = create_polynomial_model(degree)
    model.fit(X_train, y_train)
    coefficients_on_train_set = model.named_steps['linearregression'].coef_
    y_pred = model.predict(X_test)
    mse_on_test_set = mean_squared_error(y_test, y_pred)
    return name, model, mse_on_test_set, coefficients_on_train_set


# print(create_train_and_evaluate_polynomial_model(X_train, y_train, X_test, y_test))
# name, model, mse_on_test_set, coefficients_on_train_set = \
#     create_train_and_evaluate_polynomial_model(X_train, y_train, X_test, y_test)

def print_coeffs(text, model):
    if 'linear_regression' in model.named_steps.keys():
        linreg = 'linear_regression'
    else:
        linreg = 'linearregression'
    coeffs = np.concatenate(([model.named_steps[linreg].intercept_], model.named_steps[linreg].coef_[1:]))
    coeffs_str = ' '.join(np.format_float_positional(coeff, precision=4) for coeff in coeffs)
    print(text + coeffs_str)


def hyperparameter_search(X_train, y_train, X_test, y_test, from_degree=1, to_degree=15):
    degrees = range(from_degree, to_degree)
    best_degree, best_mse, best_model = None, float('inf'), None
    d_mse = {}
    for degree in degrees:
        name, model, mse_on_test_set, coefficients_on_train_set = \
            create_train_and_evaluate_polynomial_model(X_train, y_train, X_test, y_test, degree=degree)
        d_mse[degree] = mse_on_test_set
        print(f'for degree: {degree}, MSE: {mse_on_test_set}')
        if mse_on_test_set < best_mse:
            best_degree, best_mse, best_model = degree, mse_on_test_set, model
    print(f'Best degree: {best_degree}, Best MSE: {best_mse}')
    print_coeffs('Coefficients: ', best_model)
    return best_model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
            print_coeffs("Coefficients: ", model)
            mse_sum += mse
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        print(f"for degree: {degree}, MSE: {avg_mse}")
        # fit for the whole dataset
        # model, mse = train_and_evaluate_model(model, X, y, X_val, y_val)1
        model.fit(X, y)
        print_coeffs("Final Coefficients: ", model)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)
    return best_model


def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', header=None, names=['x', 'y'])
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def plot_data_and_prediction(X, y, model, title=None):
    plt.scatter(X, y, color='blue', label='Data Points')
    X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_pred)
    plt.plot(X_pred, y_pred, color='red', label='Model Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()