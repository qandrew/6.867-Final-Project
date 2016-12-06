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
import tensorflow as tf
import numpy as np
import time


print 'Loading datasets ... \n'

testdir = '/home/sitara/test_single/test' # Modify if running on your own computer
traindir = '/home/sitara/test_single/train' # Modify if running on your own computer

Xtest, Ytest = ld(testdir)
Xtrain, Ytrain = ld(traindir)

print 'Loaded training data: ', Xtrain.shape, ' ; ', Ytrain.shape,'\n'
print 'Loading testing data: ', Xtest.shape, ' ; ' , Ytest.shape, '\n'

BIN_FREQ = 23
WINDOW_SIZE = 100
NUM_CHANNELS = 3
NUM_LABELS = 11
INCLUDE_TEST_SET = False

class SpectogramConvNet:
    def __init__(self):
        '''Initialize the class by loading the required datasets
        and building the graph'''
        
        # Initializing data set
        self.train_X = Xtrain
        self.train_Y = Ytrain
        self.val_X = Xtest[:500]
        self.val_Y = Ytest[:500]
    
        if INCLUDE_TEST_SET:
            self.test_X = Xtest[500:]
            self.test_Y = Ytest[500:]
        
        self.graph = tf.Graph()
        self.define_tensorflow_graph()

    def define_tensorflow_graph(self):
        print '\n Defining model...'

        # Hyperparameters
        batch_size = 10
        learning_rate = 0.01
        layer1_filter_size = 5
        layer1_depth = 40
        layer1_stride = 1
        
        layer2_filter_size = 5
        layer2_depth = 16
        layer2_stride = 1
        
        layer3_num_hidden = 64
        layer4_num_hidden = 64
        num_training_steps = 1501

        # Add max pooling
        pooling = False
        layer1_pool_filter_size = 2
        layer1_pool_stride = 2
        layer2_pool_filter_size = 2
        layer2_pool_stride = 2

        # Enable dropout and weight decay normalization
        dropout_prob = 1.0 # set to < 1.0 to apply dropout, 1.0 to remove
        weight_penalty = 0.0 # set to > 0.0 to apply weight penalty, 0.0 to remove

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
                conv1_biases = tf.Variable(tf.zeros([layer1_depth]))
                conv1 = tf.nn.conv2d(data, conv1_weights, conv1_stride, padding='SAME', name='conv1')
                hidden = tf.nn.relu(conv1 + conv1_biases)

                # Perform local response normalization of width 5, alpha = 1e-4 and beta = 0.75
                hidden = tf.nn.local_response_normalization(hidden, depth_radius=5, bias=None, alpha=1e-4, beta=0.75)
            
                # Layer 2: Implement max pooling layer of height 3, width 4, vertical stride 1 and horizontal stride 2       
                maxpool_filter_height = 3
                maxpool_filter_width = 4
                maxpool_stride_vertical = 1
                maxpool_stride_horizontal = 2
                
                hidden = tf.nn.max_pool(hidden, ksize=[1,  maxpool_filter_height,  maxpool_filter_width, 1],
                                       strides=[1, maxpool_stride_vertical = 1, maxpool_stride_horizontal, 1],
                                        padding='SAME', name='pool1')



            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            layer3_weights = tf.Variable(tf.truncated_normal(
                [layer2_feat_map_size * layer2_feat_map_size * layer2_depth, layer3_num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))

            layer4_weights = tf.Variable(tf.truncated_normal(
              [layer4_num_hidden, NUM_LABELS], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))
            
            
                # Layer 3: Fully connected layer with 1024 units, a dropout ratio of 0.5 and ReLU non-linearities
                
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)

                # Layer 4: Fully connected layer with 1024 units, a dropout ratio of 0.5 and ReLU non-linearities
                output = tf.matmul(hidden, layer4_weights) + layer4_biases
                
                return output

            # Training computation
            logits = network_model(tf_train_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer3_weights, layer4_weights], weight_penalty)

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            batch_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(network_model(tf_test_dataset))
            train_prediction = tf.nn.softmax(network_model(tf_train_dataset))

            def train_model(num_steps=num_training_steps):
                '''Train the model with minibatches in a tensorflow session'''
                with tf.Session(graph=self.graph) as session:
                    tf.initialize_all_variables().run()
                    print 'Initializing variables...'

                    for step in range(num_steps):
                        offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
                        batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
                        batch_labels = self.train_Y[offset:(offset + batch_size), :]

                        # Data to feed into the placeholder variables in the tensorflow graph
                        feed_dict = {tf_train_batch : batch_data, tf_train_labels : batch_labels,
                                     dropout_keep_prob: dropout_prob}
                        _, l, predictions = session.run(
                          [optimizer, loss, batch_prediction], feed_dict=feed_dict)
                        if (step % 100 == 0):
                            train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
                                                                           dropout_keep_prob : 1.0})
                            val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
                            print ''
                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))

#                    # This code is for the final question
#                    if self.invariance:
#                        print "\n Obtaining final results on invariance sets!"
#                        sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X,
#                                self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X,
#                                self.inverted_val_X,]
#                        set_names = ['normal validation', 'translated', 'brightened', 'darkened',
#                                     'high contrast', 'low contrast', 'flipped', 'inverted']
#
#                        for i in range(len(sets)):
#                            preds = session.run(test_prediction,
#                                feed_dict={tf_test_dataset: sets[i], dropout_keep_prob : 1.0})
#                            print 'Accuracy on', set_names[i], 'data: %.1f%%' % accuracy(preds, self.val_Y)
#
#                            # save final preds to make confusion matrix
#                            if i == 0:
#                                self.final_val_preds = preds

            # save train model function so it can be called later
            self.train_model = train_model

def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

t1 = time.time()
conv_net = SpectogramConvNet()
conv_net.train_model()
t2 = time.time()
print "Finished training. Total time taken:", t2-t1


