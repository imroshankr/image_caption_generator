import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

def preprocess_data(x, y, limit):             #limit the deta in only two classes, preprocess_data method
    zero_index = np.where(y == 0)[0][:limit]  #we get the indices image represents 0 or 1.
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))    # stack these arrays of no. together & suffle them 
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]         # extrext only those images at these indices
    x = x.reshape(len(x), 1, 28, 28)      #reshape each images which is 28*28 pix. to a 3D block of 1*28*28 because convolutional layer takes a 3D block of data with depth as 1st dimension.
    x = x.astype("float32") / 255  # since the images contains the no. between 0 & 255  we normalizing the input by deviding each value by 255
    
    y = np_utils.to_categorical(y) #it create encoded vector from a no.  
                                     # 0-->[1,0](0 become the vector 1,0)
                                  # 1-->[0,1](1 become the vector 0,1)
    y = y.reshape(len(y), 2, 1)  # we reshape these vectors to be coloumn vectors since this is what the dence layer takes input.
               
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# actual neural network
network = [
    Convolutional((1, 28, 28), 3, 5),           #Convolutional function 5kernal, each containing matrices of size 3*3
    Sigmoid(),                      #activate the result using Sigmoid function
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),  #Reshape output in coloumn vector
    Dense(5 * 26 * 26, 100),    #pass that coloumn vector into a dense layer then mapes it to 100 units
    Sigmoid(),
    Dense(100, 2),           #mape the 100 units to 2 units
    Sigmoid()
]

# train          #remaining
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")