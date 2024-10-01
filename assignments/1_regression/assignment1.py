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

def mse_manual(X, reg_result):
    return np.c_[np.ones((X.shape[0], 1)), X.values].dot(reg_result)

def manual_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # (XT . X)^−1 . (X⊤) . y
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

def main():
    if len(sys.argv) != 3:
        print("Run file as: python assignment1.py <training.csv> <testing.csv>")
        sys.exit(1)
    
    MODEL = LinearRegression()

    train_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    X_train, y_train = preprocess(train_data)
    X_test, y_test = preprocess(test_data)

    MODEL.fit(X_train, y_train)
    
    y_train_pred = MODEL.predict(X_train)
    y_test_pred = MODEL.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f'scikit MSE train: {train_mse}')
    print(f'scikit MSE test: {test_mse}')

    regression_result = manual_regression(X_train.values, y_train)
    y_train_pred_manual = mse_manual(X_train, regression_result)
    y_test_pred_manual = mse_manual(X_test, regression_result)

    train_mse_manual = mean_squared_error(y_train, y_train_pred_manual)
    test_mse_manual = mean_squared_error(y_test, y_test_pred_manual)

    print(f'Simon MSE train: {train_mse_manual}')
    print(f'Simon MSE test: {test_mse_manual}')

if __name__ == "__main__":
    main()