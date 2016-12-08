# -*- coding: utf-8 -*-
"""
Denote by v the vector representation of an MNIST digit, with dimension 1024 x 1
Denote by w the vector representation of a spectogram, with dimension 1024 | 512 x 1

Map v to an h dimensional space with the transform W_m * v + b_m
Map w to an h dimensional space with the transform ReLU(W_d * w + b_d)

We use stochastic gradient descent with a learning rate of 1e-6 and momentum of 
0.9 across batches of 40 images to train our embedding and alignment model, and 
run our training for 20 epochs
"""


'''Given many pairs of (v,w) vectors, computes 
'''

import numpy as np
import theano
import theano.tensor as T

### SIMPLE HELPER FUNCTIONS ###
def ReLU(x):
    return max(0,x)

def create_weight(row, col, name):
    return theano.shared(value=np.zeros((row, col), dtype=theano.config.floatX),name=name, borrow=True)

def create_bias(row, name)
    return theano.shared(value=np.zeros((row,1), dtype=theano.config.floatX), name=name, borrow=True)

def similarity(y, x):
    ''' Given the h-dimensional embedding of v & w, computes the similarity of the embeddings.
        We use the model in Harwath, described below
            An overall image-caption similarity score is then computed by summing the scores of
            all words in the caption, thresholded below at 0:
            S_kl = sum_{t in G_l} max_{i in G_k} (0,y_i.T, x_t)
            where G_l denotes the set of image fragments in image l, and
            G_k is the set of word spectrograms in caption k.

        We use single fragments for each MNIST image and a single spectrogram, so this reduces to
        max(0, y.T*w).
    '''
    n1, d1 = y.shape
    n2, d2 = x.shape
    assert n1==n2 and d1==d2, 'embedded vector shapes do not match'
    
    return ReLU(np.dot(y,x))

def embed_v(W_m, b_m, v, h):
    ''' Embeds a v vector into h-dimensional space'''
    y = np.dot(W_m,v) + b_m
    assert y.shape = (h,1)
    return y
    
def embed_w(W_d, b_d, w, h):
    ''' Embeds a w vector into h-dimensional space'''
    x = ReLU(np.dot(W_d,w) + b_d)
    assert x.shape = (h,1)
    return x            
################################
    
class EmbeddedAligment(self):
        """
        EmbeddedAlignment model specification in Theano.

        v, w - vectors of MNIST and spectrogram
        h - Embedded space dimension
        """

        def __init__(self, v, w, h=512):
            
            # Initialise parameters
            self.h = h
            self.W_m = create_weight(row=, col=, name='W_m')
            self.W_d = create_weight(row= , col=, name='W_d')  
            self.b_m = create_bias(row=h, name='b_m')
            self.b_d = create_bias(row=h, name='b_d')
            
            # Initialize minibatch size
            self.batch_size = 40
            
            # Model parameters
            self.params = [self.W_m, self.b_m, self.W_d, self.b_d]
    
            # Model feature data, x
            self.v = v
            self.w = w
            
        def cost_function(v, w):
            ''' Given a pair of v and w vectors computes the cost function given
                the current parameters'''
            y = embed_v(self.W_m,self.b_m, v)
            x = embed_w(self.W_d,self.b_d, w)
            # TODO: Finish this
            
            y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
            y_pred /= y_pred.sum(axis=-1, keepdims=True)
            cce = T.nnet.categorical_crossentropy(y_pred, y_true)
            return cce

    
def stoch_grad_desc_train_model(filename, gamma=0.13, epochs=1000, B=20):
    """
    Train the logistic regression model using
    stochastic gradient descent.

    filename - File path of MNIST dataset
    gamma - Step size or "learning rate" for gradient descent
    epochs - Maximum number of epochs to run SGD
    B - The batch size for each minibatch
    """

    # Obtain the correct dataset partitions
    datasets = load_mnist_data(filename)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Calculate the number of minibatches for each of
    # the training, validation and test dataset partitions
    # Note the use of the // operator which is for 
    # integer floor division, e.g.
    # 1.0//2 is equal to 0.0
    # 1//2 is equal to 0
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // B
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // B
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // B

    # BUILD THE MODEL
    # ===============
    print("Building the logistic regression model...")

    # Create symbolic variables for the minibatch data
    index = T.lscalar()  # Integer scalar value
    v = T.vector('v')  # Feature vectors from MNIST
    w = T.vector('w')  # Feature vectors from TIDIGIT

    # Instantiate the logistic regression model and assign the cost
    embed = EmbeddedAligment(v=v, w=w, h=512)
    cost = embed.cost_function(v,w)  # This is what we minimize with SGD

    # Compiles a set of Theano functions for both the testing
    # and validation set that compute the errors on the model
    # in a particular minibatch

# TODO: FIX THIS 
    test_model = theano.function(
        inputs=[index],
        outputs=logreg.errors(y),
        givens={x: test_set_x[index * B: (index + 1) * B],
                y: test_set_y[index * B: (index + 1) * B]})

    validate_model = theano.function(
        inputs=[index],
        outputs=logreg.errors(y),
        givens={x: valid_set_x[index * B: (index + 1) * B],
                y: valid_set_y[index * B: (index + 1) * B]})
    
    # Use Theano to compute the symbolic gradients of the 
    # cost function (negative log likelihood) with respect to
    # the underlying parameters W and b
    grad_W_m = T.grad(cost=cost, wrt=embed.W_m)
    grad_b_m = T.grad(cost=cost, wrt=embed.b_m)
    grad_W_d = T.grad(cost=cost, wrt=embed.W_d)
    grad_b_d = T.grad(cost=cost, wrt=embed.b_d)


    # This is the gradient descent step. It specifies a list of
    # tuple pairs, each of which contains a Theano variable and an
    # expression on how to update them on each step.
    updates = [
        (embed.W_m, embed.W_m - gamma * grad_W_m),
        (embed.b_m, embed.b_m - gamma * grad_b_m)
        (embed.W_d, embed.W_d - gamma * grad_W_d),
        (embed.b_d, embed.b_d - gamma * grad_b_d)
    ]

    # Similar to the above compiled Theano functions, except that 
    # it is carried out on the training data AND updates the parameters
    # W, b as it evaluates the cost for the particular minibatch
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * B: (index + 1) * B],
            y: train_set_y[index * B: (index + 1) * B]
        }
    )

    # TRAIN THE MODEL
    # ===============
    print("Training the embedded alignment model...")

    # Set parameters to stop minibatch early 
    # if performance is good enough
    patience = 5000  # Minimum number of examples to look at
    patience_increase = 2  # Increase by this for new best score
    improvement_threshold = 0.995  # Relative improvement threshold   
    # Train through this number of minibatches before
    # checking performance on the validation set
    validation_frequency = min(n_train_batches, patience // 2)

    # Keep track of the validation loss and test scores
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    # Begin the training loop
    # The outer while loop loops over the number of epochs
    # The inner for loop loops over the minibatches
    finished = False
    cur_epoch = 0
    while (cur_epoch < epochs) and (not finished):
        cur_epoch = cur_epoch + 1
        # Minibatch loop
        for minibatch_index in range(n_train_batches):
            # Calculate the average likelihood for the minibatches
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (cur_epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = np.mean(validation_losses)

                # Output current validation results
                print(
                    "Epoch %i, Minibatch %i/%i, Validation Error %f %%" % (
                        cur_epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # If we obtain the best validation score to date
                if this_validation_loss < best_validation_loss:
                    # If the improvement in the loss is within the
                    # improvement threshold, then increase the 
                    # number of iterations ("patience") until the next check
                    if this_validation_loss < best_validation_loss *  \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # Set the best loss to the new current (good) validation loss
                    best_validation_loss = this_validation_loss

                    # Now we calculate the losses on the minibatches 
                    # in the test data set. Our "test_score" is the 
                    # mean of these test losses
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)

                    # Output the current test results (indented)
                    print(
                        (
                            "     Epoch %i, Minibatch %i/%i, Test error of"
                            " best model %f %%"
                        ) % (
                            cur_epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # Serialise the model to disk using pickle
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(logreg, f)
            # If the iterations exceed the current "patience"
            # then we are finished looping for this minibatch
            if iter > patience:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            "Optimization complete with "
            "best validation score of %f %%,"
            "with test performance %f %%"
        ) % (best_validation_loss * 100., test_score * 100.)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec' % (
            cur_epoch, 
            1. * cur_epoch / (end_time - start_time)
        )
    )
    print("The code ran for %.1fs" % (end_time - start_time))
