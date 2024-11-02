import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_RATIO = 0.8

def split_df(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def manual_regression(X, y):
    # using the manual regression I made for assignment 1
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

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def get_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def binary_prediction(X, weights, bias):
    # i chose for linear approach
    model = np.dot(X, weights) + bias
    y_pred = sigmoid(model)
    return [1 if i > 0.5 else 0 for i in y_pred]

def gradient_descent(X, y, train):
    iterations = 10000
    learning_rate = 0.01
    m, n = X.shape
    weights = np.ones(n) * 0.5
    bias = 0
    loss_history = []
    recent_losses = []
    recent_accuracies = []
    
    for i in range(iterations):
        current_train = train.sample(frac=TRAIN_RATIO)
        X_current_train, y_current_train = split_df(current_train)
        m = X_current_train.shape[0]
        X = np.dot(X_current_train, weights) + bias

        y_pred = sigmoid(X)
        dw = (1/m) * np.dot(y_current_train.T, (y_pred - y_current_train))
        db = (1/m) * np.sum(y_pred - y_current_train)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        loss = get_loss(y_current_train, y_pred)
        loss_history.append(loss)
        current_pred = binary_prediction(X_current_train, weights, bias)
        accuracy = np.mean(y_current_train == current_pred)
        recent_losses.append(loss)
        recent_accuracies.append(accuracy)
        
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        if len(recent_accuracies) > 100:
            recent_accuracies.pop(0)
        
        if i % 1000 == 0 and i > 0:
            avg_loss = np.mean(recent_losses)
            avg_accuracy = np.mean(recent_accuracies)
            print(f"Iteration {i}: average loss (100 samples) = {avg_loss:.4f}, average accuracy (100 samples) = {avg_accuracy:.4f}")

    return weights, bias, loss


if __name__ == "__main__":
    df = pd.read_csv("heart.csv").dropna()
    train_size = int(len(df) * TRAIN_RATIO)
    train_df = df[:train_size]
    test_df = df[train_size:]

    X_train, y_train = split_df(train_df)
    X_test, y_test = split_df(test_df)

    manual_weights = manual_regression(X_train, y_train)
    manual_y_pred = predict_manual(X_test, manual_weights)
    test_mse_manual = mse_manual(y_test, manual_y_pred)
    print("\n-------- My implementation MSE --------")
    print(f'Manual MSE Test: {test_mse_manual}')

    print("\n-------- Gradient descent --------")
    weights, bias, losses = gradient_descent(X_train, y_train, train_df)
    gd_y_pred = binary_prediction(X_test, weights, bias)
    test_mse_gd = mse_manual(y_test, gd_y_pred)
    print(f'Gradient descent MSE Test: {test_mse_manual}\n')



