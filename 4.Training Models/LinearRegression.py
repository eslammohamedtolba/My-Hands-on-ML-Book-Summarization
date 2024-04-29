# Import the depenedencies
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            # Forward -->  Y=WX+b
            Z = np.dot(X, self.weights) + self.bias
            # Backword with MSE -->  1/2n * sum((WX+b) - Y)^2
            dweights = (1/n_samples) * np.dot(X.T, (Z - Y))  # p_L/p_weights = 1/n * sum(Z - y) * X
            dbias = (1/n_samples) * np.sum(Z - Y)                # p_L/p_bias = 1/n * sum(Z - y)
            self.weights -= self.lr * dweights
            self.bias -= self.lr * dbias
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# Create regression dataset
X, Y = make_regression(n_samples = 1000, n_features = 1, noise=3, random_state = 40)
print(X.shape, Y.shape)
# Exploring dataset
plt.figure(figsize=(7,7))
plt.scatter(X, Y, color='blue', marker="*")
plt.title("Linear Regression Dataset")
plt.show()

# Split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X , Y, test_size=0.1, random_state=40)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# Create model and train it
LRModel = LinearRegression(n_iter=1000)
LRModel.fit(x_train, y_train)

# Make model predict on test data
y_pred = LRModel.predict(x_test)
# Show the first five values of actual and predicted values
print(list(zip(y_pred, y_test))[:5])
# Visualize the difference between prediction and actual values
plt.figure(figsize=(7,7))
cmap = plt.get_cmap('viridis')
m1 = plt.scatter(x_train, y_test, color=cmap(0.9), marker = "*")
m2 = plt.scatter(x_test, y_test, color = cmap(0.2),marker = "*")
plt.scatter(y_pred,color='blue',marker="x")
plt.plot(x_test,y_pred,color='black')
plt.title("the difference between prediction and actual values")
plt.show()

# Calculate the error by MSE
def FindMSE(y_pred, y_actual):
    error = np.mean((y_actual - y_pred)**2)
    return error
print("the regression error is", FindMSE(y_pred,y_test))



