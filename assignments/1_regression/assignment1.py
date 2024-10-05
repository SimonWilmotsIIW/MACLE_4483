#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

def preprocess(data):
    # X = input, y = target
    X = data.drop(columns=['medv'])
    y = data['medv'].dropna().values
    return X, y

def manual_regression(X, y):
    # add offset to make sure the linear fit does not have to pass through the origin
    # done by concatenating the X feature matrix with a columns of 1s in front
    X_offset = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    # (XT . X)^−1 . (X⊤) . y
    w = np.linalg.inv(X_offset.T.dot(X_offset)).dot(X_offset.T).dot(y)
    return w


def predict_manual(X, reg_weights):
    # compare true data to predicted weights (with extra offset also done with the manual regression)
    return np.concatenate([np.ones((X.shape[0], 1)), X.values], axis=1).dot(reg_weights)

def mse_manual(y_true, y_pred):
    # take mean of the error squared
    # mean( ∑(y − yi)^2 )
    return np.mean((y_true - y_pred) ** 2)

# I wanted to explore polynomial regression and had the following idea:
# To create the polynomial, I think I have to take the features (X) and transform them into X^2, X^3, ..., X^N with N = degree of the polynomial
# X => [1, X, X^2, X^3, ..., X^N]
#  -------------------------------------------------------------------------
# Other options (with scikit learn) are just fitting a polynomial using:
#   poly = PolynomialFeatures(degree=N)
#   X_poly = poly.fit_transform(X)
# Second example is from an example on geeksforgeeks.org

def main():
    if len(sys.argv) != 3:
        print("Run file as: python assignment1.py <training.csv> <testing.csv>")
        sys.exit(1)
    
    MODEL = LinearRegression()

    train_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    X_train, y_train = preprocess(train_data)
    X_test, y_test = preprocess(test_data)

    # fit & predict with scikit model
    MODEL.fit(X_train, y_train)
    y_train_pred = MODEL.predict(X_train)
    y_test_pred = MODEL.predict(X_test)
    # use scikit's MSE calculation
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("-------- Scikit MSE --------")
    print(f'Train: {train_mse}')
    print(f'Test: {test_mse}')

    # calculate weights with own model
    regression_weights = manual_regression(X_train.values, y_train)
    train_pred_y = predict_manual(X_train, regression_weights)
    test_pred_y = predict_manual(X_test, regression_weights)

    # first did this using scikit learn
    # tried implementing my own using the lecture's formula
    train_mse_manual = mse_manual(y_train, train_pred_y)
    test_mse_manual = mse_manual(y_test, test_pred_y)
    
    print("\n-------- My implementation MSE --------")
    print(f'Ttrain: {train_mse_manual}')
    print(f'Test: {test_mse_manual}\n')

    # actual difference between my implementation and scikitlearn is to the order of 10^(-15)
    print(f"Difference in train MSE (scikit - manual): {train_mse - train_mse_manual} ({((train_mse - train_mse_manual) / train_mse) * 100:.5f}%)")
    print(f"Difference in test MSE (scikit - manual): {test_mse - test_mse_manual} ({((test_mse - test_mse_manual) / test_mse) * 100:.5f}%)")

if __name__ == "__main__":
    main()