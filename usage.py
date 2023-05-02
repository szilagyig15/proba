import machine_learning_2 as m

coeffs = [100, 1, 0.2, 0.6]
X, y = m.generate_polinomial_data(coeffs, fromX=-5, toX=7, n_samples=500, noise=1, random_state=42)
X_train, X_test, y_train, y_test = m.train_test_split(X, y, test_size=0.2, random_state=42)

name, model, mse_on_test_set, coefficients_on_train_set = \
    m.create_train_and_evaluate_polynomial_model(X_train, y_train, X_test, y_test, degree=3)

print(coefficients_on_train_set)
best_model = m.hyperparameter_search(X_train, y_train, X_test, y_test, from_degree=1, to_degree=15)

print(1)
