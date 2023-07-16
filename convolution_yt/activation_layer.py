import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):  #two parameters actvn & it's derivative 
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):   #forward method simply apply the actvn to input
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input)) #it gives dE/dx=(dE/dY)*f'(x)