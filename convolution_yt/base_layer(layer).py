import numpy as np        #no point in using activition in dence layer
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):  #output gradient=dE/dB=(dE/dY)
        weights_gradient = np.dot(output_gradient, self.input.T)  #dE/dW=(dE/dY)*X(transpose)
        input_gradient = np.dot(self.weights.T, output_gradient)  #dE/dX=W(transpose)*dE/dY
        self.weights -= learning_rate * weights_gradient #update the parameters
        self.bias -= learning_rate * output_gradient
        return input_gradient