# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:28:40 2016

@author: sitara

FOR EACH BATCH v,w:

    STEP 1: Embed v, w to y, x in h-dimensional space
    STEP 2: Compute the cost function
    STEP 3: Perform a gradient update step on this cost
    
    REPEAT UNTIL COST FUNCTION CONVERGES
"""
import numpy as np
import theano
import os

''' DIMENSION PARAMETERS '''
n = 11
h = 512
dim_v = 1024
dim_w = 512

''' SGD PARAMETERS '''
learning_rate = 1e-3
momentum = 0.9
max_iterations = 1e4

''' INITIALISE WEIGHTS AND BIASES '''
W_v = theano.shared(np.random.normal(size=(h,dim_v)), 'W_v')
W_w = theano.shared(np.random.normal(size=(h,dim_w)), 'W_w')
b_v = theano.shared(np.random.normal(size=(h,1)), 'b_v')
b_w = theano.shared(np.random.normal(size=(h,1)), 'b_w')

""" IMPORTING DATA """
data1024dir = '../cnn_code/spectro1024Act'
data512dir = '../cnn_code/spectro512Act'
dataMnistdir = '../cnn_code/mnistAct'


# each activation layer is a row in the matrix
# and it's actual classification is in the corresponding YVal array at the same index
data1024 = np.array([[0 for i in xrange(1024)]])
data1024YVal = []
data512 = np.matrix([[0 for i in xrange(512)]])
data512YVal = []
dataMnist = np.array([[0 for i in xrange(512)]])
dataMnistYVal = []


for subdir, dirs, files in os.walk(data512dir):
    for file in files:
        print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".txt"):
            # print filepath
            temp = np.loadtxt(open(filepath))
            # print temp.shape
            i = filepath[-5]
            if i == 'z': j = 10
            elif i == 'o': j = 0
            else: j = int(i)
            for i in xrange(temp.shape[0]):
              data512YVal.append(j)
            data512 = np.concatenate((data512,temp))
data512YVal = np.matrix(data512YVal).T
data512 = data512[1:]
# print data512.shape
# print data512YVal.shape            

for subdir, dirs, files in os.walk(data1024dir):
    for file in files:
        print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".txt"):
            # print filepath
            temp = np.loadtxt(open(filepath))
            # print temp.shape
            i = filepath[-5]
            if i == 'z': j = 10
            elif i == 'o': j = 0
            else: j = int(i)
            for i in xrange(temp.shape[0]):
              data1024YVal.append(j)
            data1024 = np.concatenate((data1024,temp))
data1024YVal = np.matrix(data1024YVal).T
data1024 = data1024[1:]
# print data1024.shape
# print data1024YVal.shape  

for subdir, dirs, files in os.walk(dataMnistdir):
    for file in files:
        print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".txt"):
            # print filepath
            temp = np.loadtxt(open(filepath))
            # print temp.shape
            i = filepath[-5]
            if i == 'z': j = 10
            elif i == 'o': j = 0
            else: j = int(i)
            for i in xrange(temp.shape[0]):
              dataMnistYVal.append(j)
            dataMnist = np.concatenate((dataMnist,temp))
dataMnistYVal = np.matrix(dataMnistYVal).T
dataMnist = dataMnist[1:]
# print dataMnist.shape
# print dataMnistYVal.shape  

''' EMBEDDING MODEL '''

V = theano.tensor.matrix('V') 
W = theano.tensor.matrix('W') 

V_vals = np.random.random((n,dim_v))
W_vals = np.random.random((n,dim_w))

#Cast data to float32
V_vals=np.array(V_vals, dtype=np.float32)
W_vals=np.array(W_vals, dtype=np.float32)


print '----Input v, w shape --------'
print V_vals.shape
print W_vals.shape
print '-----------------------------'


Y = (theano.tensor.dot(W_v,V.T) + theano.tensor.tile(b_v, reps=(1,n))).T # Creates a function that evaluates the new embedding of V.
X = (theano.tensor.dot(W_w,W.T) + theano.tensor.tile(b_w, reps=(1,n))).T # Creates a function that evaluates the new embedding of W
embed_Y = theano.function([V], Y, name='compute Y from V')  ##called like compute_Y(V_vals) 
embed_X = theano.function([W], X, name='compute X from W')  ##called like compute_X(W_vals) 


# Compute the cost of an embedding

cost = theano.tensor.nnet.relu(1+ theano.tensor.dot((theano.tensor.dot(W_v,V.T) + theano.tensor.tile(b_v, reps=(1,n))).T,theano.tensor.dot(W_w,W.T) + theano.tensor.tile(b_w, reps=(1,n))) - theano.tensor.nlinalg.diag(theano.tensor.dot((theano.tensor.dot(W_v,V.T) + theano.tensor.tile(b_v, reps=(1,n))).T,theano.tensor.dot(W_w,W.T) + theano.tensor.tile(b_w, reps=(1,n))))).sum()

#cost_calc = theano.tensor.nnet.relu(1+ theano.tensor.dot(compute_Y(V),compute_X(W).T) - theano.tensor.nlinalg.diag(theano.tensor.dot(compute_Y(V),compute_X(W).T))).sum()

gW_v, gb_v, gW_w, gb_w = theano.tensor.grad(cost, [W_v, b_v, W_w, b_w])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[V,W],
          outputs=[cost],
          updates=((W_v, W_v - learning_rate * gW_v + (1. - momentum)*gW_v),(b_v, b_v - learning_rate * gb_v + (1. - momentum)*gb_v),(W_w, W_w - learning_rate * gW_w + (1. - momentum)*gW_w), (b_w, b_w - learning_rate * gb_w + (1. - momentum)*gb_w)))

# Train

thresh = 1e-4
iters = 0
cost_history = [np.inf]
converged = False
while iters < max_iterations and not converged:
    # Grab a random pair of batch data. Must have 11 rows and the corresponding dimensions for MNIST and TIDIGITS
    #V_vals = 
    #W_vals = 
    
    V_vals = np.random.normal(size=(11,1024))    
    W_vals = np.random.normal(size=(11,512))
    
    iters += 1
    cost = train(V_vals,W_vals)[0]
    print cost
    if abs(cost-cost_history[-1]) < thresh:
        converged = True
    cost_history.append(cost)
    if iters%100==0:
        print iters,' : ',cost
print("Final model:")
print(W_v.get_value())
print(b_v.get_value())
print(W_w.get_value())
print(b_w.get_value())