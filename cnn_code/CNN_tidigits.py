# -*- coding: utf-8 -*-
"""
Beginning to construct tensorflow architecture for classifying tidigits 
spectograms.

Architecture based on the following paper by David Harwath:
        http://people.csail.mit.edu/dharwath/papers/Harwath_ASRU-15.pdf
    
Specifically, they describe their methodology as follows:
        1. Pixel-by-pixel mean image spectrogram subtraction,
        with the mean spectrogram estimated over the entire
        training set;
        
        2. Convolutional layer with filters sized 5 frames by 40
        features with a stride of 1, vertical padding of 1 pixel on
        both the top and bottom, and 64 output channels with a
        ReLU nonlinearity;
        
        3. Local response normalization of width 5, α = 0.0001,
        and β = 0.75;
        
        4. Max pooling layer of height 3, width 4, vertical stride
        1, and horizontal stride 2;
        
        5. Two fully connected layers of 1024 units each, with a
        dropout ratio of 0.5 and ReLU nonlinearities;
        
        6. A softmax classification layer

    To extract vector representations for each word in some
    image caption, we feed the word’s spectrogram through the
    network and discard the softmax outputs, retaining only the
    activations of the dW = 1024 dimensional fully connected
    layer immediately before the classification layer.

CONV -> MAX POOL -> FULLY -> FULLY ->SOFTMAX
"""

from load_data import get_data as ld
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


predictions = np.array([1,0,0])
labels =np.array([1,0,0])
#print accuracy(predictions, labels)



print 'Loading datasets ... \n'

testdir = '/home/sitara/test_single/test' # Modify if running on your own computer
traindir = '/home/sitara/test_single/train' # Modify if running on your own computer

#testdir = '/home/andrew/Dropbox (MIT)/6867_Project/single_utterances/test' #Andrew
#traindir = '/home/andrew/Dropbox (MIT)/6867_Project/single_utterances/train' #Andrew

Xtest, Ytest = ld(testdir)
Xtrain, Ytrain = ld(traindir)

def get_digit(digit_indexes, X, Y):
    x_vals = []
    y_vals = []
    for i in range(len(X)):
        y = Y[i]
        for digit_index in digit_indexes:
            if y[digit_index] > 0:
#                print digit_index
#                plt.imshow(X[i], aspect='auto', interpolation='none')
#                plt.show()
                x_vals.append(X[i])
                y_val = np.zeros((len(digit_indexes),1))
                y_val[digit_index-min(digit_indexes)] = 1
                y_vals.append(y_val)
                break
    return np.array(x_vals, dtype=np.float32), np.array(y_vals, dtype=np.float32)
            
def make_bw(X,Y):
    for i in range(len(X)):
        if Y[i][0]==1:
            X[i] = 0
        else:
            X[i] = 10
    return np.array(X), Y
            
Xtrain,Ytrain = get_digit([7,8],Xtrain,Ytrain)
Xtest,Ytest = get_digit([7,8],Xtest,Ytest)

Xtrain, Ytrain = make_bw(Xtrain, Ytrain)
Xtest, Ytest = make_bw(Xtest, Ytest)

# Reshape input vectors for one input channel
Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2],1)
Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2],1)

# Reshape labels to (num_data,11,)
Ytrain = Ytrain.reshape(Ytrain.shape[0],Ytrain.shape[1],)
Ytest = Ytest.reshape(Ytest.shape[0],Ytest.shape[1],)

print 'Loaded training data: ', Xtrain.shape, ' ; ', Ytrain.shape
print 'Loading testing data: ', Xtest.shape,  ' ; ', Ytest.shape

BIN_FREQ = 23
WINDOW_SIZE = 100
NUM_CHANNELS = 1
NUM_LABELS = 2
INCLUDE_TEST_SET = True

normalization = True #choose whether we want local response normalization or not

print '\nSettings:'
print 'BIN_FREQ', BIN_FREQ
print 'WINDOW_SIZE', WINDOW_SIZE
print "NUM_CHANNELS", NUM_CHANNELS
print "NUM_LABELS", NUM_LABELS
print "INCLUDE_TEST_SET", INCLUDE_TEST_SET
print "NORMALIZATION", normalization

class SpectogramConvNet:
    def __init__(self):
        '''Initialize the class by loading the required datasets
        and building the graph'''
        
        # Initializing data set
        self.train_X = Xtrain #[:1500]
        self.train_Y = Ytrain #[:1500]

        self.val_X = Xtest[:400]
        self.val_Y = Ytest[:400]
        if INCLUDE_TEST_SET:
            self.test_X = Xtest[500:600]
            self.test_Y = Ytest[500:600]
        
        self.graph = tf.Graph()
        self.define_tensorflow_graph()

    def define_tensorflow_graph(self):
        print '\nDefining model...'

        # Hyperparameters
        batch_size = 10
        learning_rate = 1e-3
        num_training_steps = 15000

        with self.graph.as_default():
            # Input data
            tf_train_batch = tf.placeholder(
                tf.float32, shape=(batch_size, WINDOW_SIZE, BIN_FREQ, NUM_CHANNELS))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.placeholder(
                tf.float32, shape=[len(self.val_X),  WINDOW_SIZE, BIN_FREQ, NUM_CHANNELS])
            tf_train_dataset = tf.placeholder(
                tf.float32, shape=[len(self.train_X),  WINDOW_SIZE, BIN_FREQ, NUM_CHANNELS])
 
            full2_keep_prob = tf.placeholder(tf.float32)
            full1_keep_prob = tf.placeholder(tf.float32)


            # Layer 3 Params:


            # Layer 4 Params:
            full2_depth = 1024                              
            full2_weights = [full2_depth, NUM_LABELS]
            full2_weights = tf.Variable(tf.truncated_normal(full2_weights, stddev=0.1))    
            full2_bias = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))
            # Model
            def network_model(data):
                '''Define the actual network architecture'''

                # Layer 1 is a convolutional neural net constructed using 
                    #tf.nn.conv2d(input, filter, strides, padding, name=None)
                    #Given an input tensor of shape [batch, in_height, in_width, in_channels] 
                    #and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
               
                conv1_weights = [5,BIN_FREQ,NUM_CHANNELS,64]         # [filter_height, filter_width, in_channels, output_channels]
                conv1_weights = tf.Variable(tf.truncated_normal(conv1_weights, stddev=0.1))
                conv1_stride = [1,1,1,1]                             # [1, stride, stride, 1]
                conv1_biases = tf.Variable(tf.zeros([64]))
                conv1 = tf.nn.conv2d(data, conv1_weights, conv1_stride, padding='SAME', name='conv1')
                hidden = tf.nn.relu(conv1 + conv1_biases)

                # Perform local response normalization of width 5, alpha = 1e-4 and beta = 0.75
                if normalization:
                    hidden = tf.nn.local_response_normalization(hidden, depth_radius=5, bias=None, alpha=1e-4, beta=0.75)
            
                # Layer 2: Implement max pooling layer of height 3, width 4, vertical stride 1 and horizontal stride 2       
                maxpool_filter_height = 3
                maxpool_filter_width = 4
                maxpool_stride_vertical = 1
                maxpool_stride_horizontal = 2
                
                hidden = tf.nn.max_pool(hidden, ksize=[1,  maxpool_filter_height,  maxpool_filter_width, 1],
                                       strides=[1, maxpool_stride_vertical, maxpool_stride_horizontal, 1],
                                        padding='SAME', name='pool1')
            
                # Layer 3: Fully connected layer with 1024 units, a dropout ratio of 0.5 and ReLU non-linearities
                full1_depth = 1024
                
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                print 'A', shape[1], shape[2], shape[3]
                full1_weights = [shape[1] * shape[2] * shape[3], full1_depth]
                full1_weights = tf.Variable(tf.truncated_normal(full1_weights, stddev=0.1))    
                full1_bias = tf.Variable(tf.constant(1.0, shape=[full1_depth]))
                
                hidden = tf.nn.relu(tf.matmul(reshape, full1_weights) + full1_bias)
                hidden = tf.nn.dropout(hidden, full1_keep_prob)

                # Layer 4: Fully connected layer with 1024 units, a dropout ratio of 0.5 and ReLU non-linearities
                hidden = tf.nn.relu(tf.matmul(hidden, full2_weights) + full2_bias)      
                hidden = tf.nn.dropout(hidden, full2_keep_prob)
          
          
                return hidden

            # Training computation
            logits = network_model(tf_train_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            batch_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(network_model(tf_test_dataset))
            train_prediction = tf.nn.softmax(network_model(tf_train_dataset))
           
            def train_model(num_steps=num_training_steps):
                '''Train the model with minibatches in a tensorflow session'''
                dropout_prob = 1 # No dropout
                with tf.Session(graph=self.graph) as session:
                    tf.global_variables_initializer().run()
                    print '\nInitializing variables...'

                    for step in range(num_steps):
                        offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
                        batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
                        batch_labels = self.train_Y[offset:(offset + batch_size), :]
                        # Data to feed into the placeholder variables in the tensorflow graph
                        
                        feed_dict = {tf_train_batch : batch_data, tf_train_labels : batch_labels,
                                     full1_keep_prob: dropout_prob, full2_keep_prob: dropout_prob}
                        _, l, predictions = session.run(
                          [optimizer, loss, batch_prediction], feed_dict=feed_dict)
                        if (step % 100 == 0):
                            train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
                                                                           full1_keep_prob : dropout_prob, full2_keep_prob : dropout_prob})
                            val_preds = session.run(valid_prediction, feed_dict={full1_keep_prob : dropout_prob, full2_keep_prob : dropout_prob})

                            
                            print ''
                            print('\tBatch loss at step %d: %f' % (step, l))
                            print('\tBatch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print predictions, batch_labels
                            print('\tValidation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('\tFull train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))

            # save train model function so it can be called later
            self.train_model = train_model


t1 = time.time()
conv_net = SpectogramConvNet()
conv_net.train_model()
t2 = time.time()
print "Finished training. Total time taken:", t2-t1


