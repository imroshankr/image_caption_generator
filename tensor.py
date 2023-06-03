import numpy as np      #import
import tensorflow as tf 
# end1*****************************************************************

w=tf.variable(0, dtype=tf.float32)                   #define the parameter initilize with 0 and typye of floating point
optimizer = tf.keras.optimizers.adam(0.1)          #using adam optimazition algorithim and learning rate is 0.1

def train_step():         #define a single traine step
with tf.gradientTape() as tape: #to record the order of the seq of order of the sequence of opration needed to compute the cost funn-forwardprp (#we only have to do forward prop that mean omly have to write the code to compute the value of the cost funn and backprop done automatically by this in tensorflow.)
        cost = w**2-10*w+25                       #cost function 
    trainable_variables= [W]
    grads =tape.gradient(cost , trainable_variables)        #compute gradient 
    optimizer.applu_gradients(zip(grads, trainable_variables))         #use optimizer to apply gradients and gradients are grads, trainable_variables and we use zip funn to take the list of gradient and pair them up
print (W)      #output **numpy=0.0**
# end2*****************************************************************

train_step()
print(W)       #output **numpy=0.09999997**
# end3******************************************************************

for i in range(1000)
    train_step()
    print(W)             #output **numpy=5.000001**(nearly equals to the minimum of the cost funn)
# end4******************************************************************





#code for cost funn depends on x and parameter w
w=tf.variable(0, dtype=tf.float32) 
x=np.array([1.0,-10.0, 25.0], dtype=np)    #difference(x is coffiecient of the cost funn)
optimizer = tf.keras.optimizers.adam(0.1)
def training(x,W , optimizer):
    def cost_fn():
        return x[0]*W**2-x[1]*W+x[2]   # cost funn
    for i in range(1000):
        optimizer.minimize(cost_fn, [W]      
        return W    
W=training(x, W, optimizer)
                           print(w)      ##output--5.000001
                           
      
#****************************************************************************
    





    




    