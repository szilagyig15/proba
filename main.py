import machine_learning_2 as m

coeffs = [100, 1, 0.2]
X, y = m.generate_polinomial_data(coeffs, fromX=-5, toX=7, n_samples=500, noise=1, random_state=42, filepath='data.csv')
X_train, X_test, y_train, y_test = m.train_test_split(X, y, test_size=0.2, random_state=42)



tomb=[]

for d in range(20):
    name, model, mse_on_test_set, coefficients_on_train_set = m.creat_train_and_test_evaluate_polynomial_model(X_train,
                                                                                                               X_test,
                                                                                                               y_train,
                                                                                                               y_test,
                                                                                                               degree=d)
    tomb.append(mse_on_test_set)

print(tomb)
print(tomb.index(min(tomb)))



print('alma')
