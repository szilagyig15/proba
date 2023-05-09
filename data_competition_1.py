from machine_learning_2 import cross_validate, load_data, plot_data_and_prediction

X, y = load_data("data_competition2_train.csv")
model = cross_validate(X, y)
plot_data_and_prediction(X, y, model)
