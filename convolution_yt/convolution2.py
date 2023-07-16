import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):   #constructor takes n3 parameters, kernal_size-size of each matrix inside each kerna, depth-how many kernal we want.
        input_depth, input_height, input_width = input_shape       ##unpacking the input shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)         #compute the output shape it has 3 dimensions,(1).depth=no. of kernals,(2)hight and (3)the width of each output matrix
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)  #shape of the kernal which is 4 dimensional because we have multiple kernal and each have 3 dimension block. here depth= no. of kernals, input_depth=depth of the each kernal. kernal_size= sixe of each mtrices in kernal
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)             # we used scipy to compute the cross corelation   

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)     #we start by initilizing empety arrays for the kernal gradient and the input gradient.
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):          #we compute the derivative of E w.r.t. Kij, 
            for j in range(self.input_depth):         
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")     #valid correlation
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")       # full convolution

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient