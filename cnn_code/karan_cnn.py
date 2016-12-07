import tensorflow as tf
import numpy as np
import sys
import math
import time
from load_data import get_data as ld
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

testdir = '/Users/karan/research/single_utterances/test'
traindir = '/Users/karan/research/single_utterances/train'

testdir = '/home/andrew/Dropbox (MIT)/6867_Project/single_utterances/test' #Andrew
traindir = '/home/andrew/Dropbox (MIT)/6867_Project/single_utterances/train' #Andrew


IMAGE_HEIGHT = 100
IMAGE_WIDTH = 23
NUM_CHANNELS = 1
NUM_LABELS = 11
INCLUDE_TEST_SET = True
VALIDATION_SIZE = 1000  # Size of the validation set.

class SpecConvNet:
    def __init__(self):
        '''Initialize the class by loading the required datasets
        and building the graph'''
        self.load_dataset(traindir,testdir)
        self.graph = tf.Graph()
        self.define_tensorflow_graph()

        #for plotting
        self.batch_plot = []
        self.val_plot = []
        self.train_plot = []
        self.steps = []

    def extract_data(self,traindir, testdir):
        print 'Loading datasets ... \n'

        Xtest, Ytest = ld(testdir)
        Xtrain, Ytrain = ld(traindir)

        # Reshape input vectors for one input channel
        Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2],NUM_CHANNELS)
        Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2],1)

        # Reshape labels to (num_data,11,)
        Ytrain = Ytrain.reshape(Ytrain.shape[0],Ytrain.shape[1],)
        Ytest = Ytest.reshape(Ytest.shape[0],Ytest.shape[1],)

        print 'Loaded training data: ', Xtrain.shape, ' ; ', Ytrain.shape
        print 'Loading testing data: ', Xtest.shape, ' ; ' , Ytest.shape

        return Xtrain, Xtest, Ytrain, Ytest

    def load_dataset(self,traindir,testdir):
        Xtrain, Xtest, Ytrain, Ytest = self.extract_data(traindir, testdir)
        self.train_X = Xtrain
        self.train_Y = Ytrain
        self.val_X = Xtest[:VALIDATION_SIZE]
        self.val_Y = Ytest[:VALIDATION_SIZE]

        if INCLUDE_TEST_SET:
            self.test_X = Xtest[VALIDATION_SIZE:]
            self.test_Y = Ytest[VALIDATION_SIZE:]

        print 'Training set', self.train_X.shape, self.train_Y.shape
        print 'Validation set', self.val_X.shape, self.val_Y.shape
        if INCLUDE_TEST_SET: print 'Test set', self.test_X.shape, self.test_Y.shape

    def define_tensorflow_graph(self):
        print '\nDefining model...'
        # Hyperparameters
        batch_size = 10
        learning_rate = 0.001
        layer1_filter_height = 5
        layer1_filter_width = 23
        layer1_depth = 64
        layer1_stride = 1

        layer2_num_hidden = 1024
        layer3_num_hidden = 1024

        num_training_steps = 1501 #1501
        print 'doing', num_training_steps, 'training steps'

        # Add max pooling
        pooling = True
        layer1_pool_filter_height = 3
        layer1_pool_filter_width = 4
        layer1_pool_stride_vert = 1
        layer1_pool_stride_hor = 2

        # Enable dropout and weight decay normalization
        dropout_prob = 0.5 # set to < 1.0 to apply dropout, 1.0 to remove
        weight_penalty = 0.0 # set to > 0.0 to apply weight penalty, 0.0 to remove

        with self.graph.as_default():
            # Input data
            tf_train_batch = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.placeholder(tf.float32, shape=[len(self.val_X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
            tf_train_dataset = tf.placeholder(tf.float32, shape=[len(self.train_X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Network weights/parameters that will be learned
            layer1_weights = tf.Variable(tf.truncated_normal([layer1_filter_height, layer1_filter_width, NUM_CHANNELS, layer1_depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([layer1_depth]))
            layer1_feat_map_vert = int(math.ceil(float(IMAGE_HEIGHT) / layer1_stride))
            layer1_feat_map_hor = int(math.ceil(float(IMAGE_WIDTH) / layer1_stride))

            if pooling:
                layer1_feat_map_vert = int(math.ceil(float(layer1_feat_map_vert) / layer1_pool_stride_vert))
                layer1_feat_map_hor = int(math.ceil(float(layer1_feat_map_hor) / layer1_pool_stride_hor))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer1_feat_map_vert * layer1_feat_map_hor * layer1_depth, layer2_num_hidden], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_num_hidden]))

            layer3_weights = tf.Variable(tf.truncated_normal(
              [layer3_num_hidden, NUM_LABELS], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

            # Model
            def network_model(data):
                '''Define the actual network architecture'''

                # Layer 1
                conv1 = tf.nn.conv2d(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
                hidden = tf.nn.relu(conv1 + layer1_biases)

                if pooling:
                    hidden = tf.nn.max_pool(hidden, ksize=[1, layer1_pool_filter_height, layer1_pool_filter_width, 1],
                                       strides=[1, layer1_pool_stride_vert, layer1_pool_stride_hor, 1],
                                       padding='SAME', name='pool1')

                # Layer 2
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)

                # Layer 3
                output = tf.matmul(hidden, layer3_weights) + layer3_biases
                return output

            # Training computation
            logits = network_model(tf_train_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer2_weights, layer3_weights], weight_penalty)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
                            batch_acc = accuracy(predictions, batch_labels)
                            val_acc = accuracy(val_preds, self.val_Y)
                            train_acc = accuracy(train_preds, self.train_Y)

                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % batch_acc)
                            print('Validation accuracy: %.1f%%' % val_acc)
                            print('Full train accuracy: %.1f%%' % train_acc)

                            self.batch_plot.append(batch_acc)
                            self.val_plot.append(val_acc)
                            self.train_plot.append(train_acc)
                            self.steps.append(step)

            # save train model function so it can be called later
            self.train_model = train_model

def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == '__main__':
    t1 = time.time()
    conv_net = SpecConvNet()
    conv_net.train_model()
    t2 = time.time()
    print "Finished training. Total time taken:", t2-t1

    print "Plotting"
    plt.plot(conv_net.steps,conv_net.batch_plot,'ro-',label='batch')
    plt.plot(conv_net.steps,conv_net.val_plot,'go-',label='validation')
    plt.plot(conv_net.steps,conv_net.train_plot,'ro-',label='train')
    plt.legend()
    plt.show()
