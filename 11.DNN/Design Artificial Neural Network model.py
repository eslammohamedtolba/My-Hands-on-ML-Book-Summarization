# Designing Artificial Neural Network
'''
input   firstlayer      output

        neuron1
X1
        neuron2
X2                          y
        neuron3
X3
        neuron4
'''

import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    # (1/(1-e^-z))*(1 - 1/(1-e^-z))
    return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))


class ArtificialNeuralNetwork:
    def __init__(self):
        print("Artificial Neural Network has been created")
        
    def forward(self):
        # first layer
        self.z1 = np.dot(self.input,self.Weightsinput)
        self.a = sigmoid(self.z1) 
        
        # output(second layer)
        self.z2 = np.dot(self.a,self.WeightfirstLayer)
        self.y_pred = sigmoid(self.z2)
    def backward(self):
        # learning rate = 1
        # j = (y_true - y_pred)^2    && y_pred = sigmoid(z)  && z = wx+b
        # p_j/p_w2 = (p_j/p_y_pred)      * (p_y_pred/p_z)        * (p_z2/p_w2 )
        # p_j/p_w2 = -2(y_true - y_pred) * sigmoid_derivative(z2) * a1
        # Wnew = Wold - learning_rate * p_j/p_wold
        self.WeightfirstLayer += np.dot(self.a.T, 2*(self.y_true - self.y_pred) * sigmoid_derivative(self.z2)) 
        
        # learning rate = 1
        # j = (y_true - y_pred)^2    && y_pred = sigmoid(z)  && z = wx+b
        # p_j/p_w1 = (p_j/p_y_pred)      * (p_y_pred/p_z2)        * (p_z2/p_a ) * (p_a/p_z1) * (p_z1/p_w)
        # p_j/p_w = -2(y_true - y_pred) * sigmoid_derivative(z2) * weightsoutput * sigmoid_derivative(z1) *x
        self.Weightsinput += np.dot(self.input.T,np.dot(2*(self.y_true - self.y_pred) * sigmoid_derivative(self.z2),self.WeightfirstLayer.T)*sigmoid_derivative(self.z1))
    
    def fit_ANN(self, x, y, NeuronsLayers = 4):
        self.input=x
        self.Weightsinput = np.random.rand(self.input.shape[1],NeuronsLayers)
        self.WeightfirstLayer = np.random.rand(NeuronsLayers,1)
        self.y_true=y
    
    def compile_ANN(self, epochs = 1000):
        for i in range(epochs):
            self.forward()
            self.backward()
            
    def predict(self, X):
        # first layer
        Z1 = np.dot(X,self.Weightsinput)
        A1 = sigmoid(Z1) 
        # output(second layer)
        Z2 = np.dot(A1,self.WeightfirstLayer)
        prediction = sigmoid(Z2)
        return prediction
        
        
        
'''   
# input
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
# output
Y = np.array([[0],
              [1],
              [1],
              [0]])
'''  
X = np.array([[0,0,1],
              [0,1,0],
              [1,0,0],
              [0,1,1],
              [1,0,1],
              [1,1,0],
              [1,1,1]])

Y = np.array([[0],
              [0],
              [0],
              [1],
              [1],
              [1],
              [0]])

NN = ArtificialNeuralNetwork()
NN.fit_ANN(X, Y, 5)
NN.compile_ANN(epochs = 1000)
print(NN.y_pred)



X_test = np.array([[1,1,1],  # y_test =  0
                   [1,0,1]]) # y_test =  1
NN.predict(X_test)









