import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)

# Splits dataset into training set and test set
diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(
    diabetes_X, diabetes_Y, test_size=0.33, random_state=1)

# Creates linear regression model object
linear_reg = linear_model.LinearRegression()

# Trains the model using the training sets
linear_reg.fit(diabetes_X_train, diabetes_Y_train)

# Prediction based on trained model
diabetes_Y_predict = linear_reg.predict(diabetes_X_test)

# Determine the mean squared error
print("Mean Squared Error: %.2f" % mean_squared_error(
    diabetes_Y_test, diabetes_Y_predict))

# Determine coeffificent of determination, 1 is 100% accuracy
print("Coefficent of determination: %.2f" %
      r2_score(diabetes_Y_test, diabetes_Y_predict))
