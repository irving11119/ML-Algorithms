import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


def activation_function(Y_array):
    i = 0
    for prediction in Y_array:

        prediction = round(prediction)

        if prediction < 0:
            prediction = 0
        elif prediction > 2:
            prediction = 2

        Y_array[i] = prediction
        i += 1

    return Y_array


def linear_classification():
    # Load Iris dataset
    iris_X, iris_Y = datasets.load_iris(return_X_y=True)

    # Split dataset into train and test partitions
    iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(
        iris_X, iris_Y, test_size=0.33, random_state=42)

    # Create linear classification model object
    linear_class = linear_model.LinearRegression()

    # Trains the model using linear regression model
    linear_class.fit(iris_X_train, iris_Y_train)

    # Prediction based on trained model
    iris_Y_predict_train = linear_class.predict(iris_X_train)
    iris_Y_predict_test = linear_class.predict(iris_X_test)

    # Activation function to convert regression result to multiclass regression
    iris_Y_predict_train = activation_function(iris_Y_predict_train)
    iris_Y_predict_test = activation_function(iris_Y_predict_test)

    # Determine Train Accuracy
    total_results_train = iris_Y_predict_train.shape[0]
    correct_predictions = 0
    for i in range(0, total_results_train):
        if iris_Y_predict_train[i] == iris_Y_train[i]:
            correct_predictions += 1

    train_accuracy = (correct_predictions / total_results_train) * 100.0

    # Determine Test Accuracy
    total_results_test = iris_Y_predict_test.shape[0]
    correct_predictions = 0
    for i in range(0, total_results_test):
        if iris_Y_predict_test[i] == iris_Y_test[i]:
            correct_predictions += 1

    test_accuracy = (correct_predictions / total_results_test) * 100.0

    print("Train Accuracy: %.2f" % train_accuracy)
    print("Test Accuracy: %.2f" % test_accuracy)


if __name__ == "__main__":
    linear_classification()
