import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from machine_learning import generate_synthetic_data


def fit_linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    return lr
