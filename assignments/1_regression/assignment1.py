import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

def preprocess(data):
    # X = input, y = target
    X = data.drop(columns=['medv']).dropna()
    y = data['medv'].dropna().values
    return X, y

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def manual_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    result = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return result

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
    
    # y_train_pred = MODEL.predict(X_train)
    # y_test_pred = MODEL.predict(X_test)

    # train_mse = mse(y_train, y_train_pred)
    # test_mse = mse(y_test, y_test_pred)

    # print(f'Train MSE (scikit): {train_mse}')
    # print(f'Test MSE (scikit): {test_mse}')

    # theta_best = manual_regression(X_train.values, y_train)
    # y_train_pred_manual = np.c_[np.ones((X_train.shape[0], 1)), X_train.values].dot(theta_best)
    # y_test_pred_manual = np.c_[np.ones((X_test.shape[0], 1)), X_test.values].dot(theta_best)

    # train_mse_manual = mse(y_train, y_train_pred_manual)
    # test_mse_manual = mse(y_test, y_test_pred_manual)

    # print(f'Train MSE (manual): {train_mse_manual}')
    # print(f'Test MSE (manual): {test_mse_manual}')

if __name__ == "__main__":
    main()