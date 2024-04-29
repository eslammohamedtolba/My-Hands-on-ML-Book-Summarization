# Import the dependencies
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

class LogisticRegression:
    def __init__(self, lr = 0.01, n_iter = 100):
        self.learning_rate = lr
        self.iteration = n_iter
        self.Weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.Weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iteration):    
            # Forward  --> Y_hate(sigmoid(WX+b))
            self.Z = np.dot(X, self.Weights) + self.bias
            self.Y_hate = sigmoid(self.Z)

            # Backward -->  loss function = 1/n*Sum(ylog(y_hate + eps)+(1-y)log(1-y_hate))
            dweights = np.dot(X.T, (self.Y_hate - Y)) / n_samples 
            dbias = np.sum(self.Y_hate - Y) / n_samples
            self.Weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias

    def predict(self,X):
        Z = np.dot(X,self.Weights) + self.bias
        y_pred = sigmoid(Z)

        # Estimating probabilities
        threshold = 0.5
        class_prediction =  [1 if y >= threshold else 0 for y in y_pred]
        return class_prediction





# Create classification dataset
X , Y = make_classification(n_samples=10000, n_features=5, random_state=40)
print(X.shape,Y.shape)
# Split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.08, random_state=40)

# Explore the data
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# Create model and train it
LRModel = LogisticRegression(n_iter=5000)
LRModel.fit(x_train, y_train)
# Make the model predict on the test data
y_pred = LRModel.predict(x_test)

# Create a confusion matrix and classification_report
con_mat = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(con_mat, cmap = "Blues", annot=True, cbar=True)
plt.show()
class_repo = classification_report(y_pred,y_test)
print(class_repo)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)
print("the model accuracy is",accuracy(y_pred,y_test),"%")




