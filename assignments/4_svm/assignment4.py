import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

TRAIN_RATIO = 0.8

#plotting function created with help of ChatGPT:
def plot_decision_boundary(ax, model, X, y, title):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the label for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and data points
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('yellow', 'black')))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(('yellow', 'black')))
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

def mse_manual(y_true, y_pred):
    # take mean of the error squared
    # mean( ∑(y − yi)^2 )
    return np.mean((y_true - y_pred) ** 2)

def split_df(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

if __name__ == "__main__":
    df = pd.read_csv("moonDataset.csv")
    train_size = int(len(df) * TRAIN_RATIO)

    train_df = df[:train_size]
    test_df = df[train_size:]

    X_train, y_train = split_df(train_df)
    X_test, y_test = split_df(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #linear SVM
    lin_svm = SVC(kernel='linear')
    lin_svm.fit(X_train, y_train)
    y_pred_linear = lin_svm.predict(X_test)
    print(f"Linear: {mse_manual(y_test, y_pred_linear)}")


    #non-linear kernels to try: poly, rbf
    nonlin_svm = SVC(kernel='poly')
    #nonlin_svm = SVC(kernel='rbf')
    nonlin_svm.fit(X_train, y_train)
    y_pred_nonlin = nonlin_svm.predict(X_test)
    print(f"Linear: {mse_manual(y_test, y_pred_nonlin)}")

    #only using X1 & X2 for vizualisation
    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    lin_svm_2d = SVC(kernel='linear').fit(X_train_2d, y_train)
    nonlin_svm_2d = SVC(kernel='poly').fit(X_train_2d, y_train)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(ax1, lin_svm_2d, X_test_2d, y_test, "Linear SVM kernel")
    plot_decision_boundary(ax2, nonlin_svm_2d, X_test_2d, y_test, "Non-linear SVM kernel")
    plt.show()