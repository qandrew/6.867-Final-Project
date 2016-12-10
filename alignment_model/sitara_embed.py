# -*- coding: utf-8 -*-
"""
Denote by V the vector representation of an MNIST digit, with dimension 1024 x n
Denote by W the vector representation of a spectogram, with dimension 1024 | 512 x n

Here, n is the number of data points.

Map V to an h dimensional space with the transform W_m * v + b_m
Map W to an h dimensional space with the transform ReLU(W_d * w + b_d)

We use stochastic gradient descent with a learning rate of 1e-6 and momentum of 
0.9 across batches of 40 images to train our embedding and alignment model, and 
run our training for 20 epochs
"""


'''Given many pairs of (v,w) vectors, computes 
'''

import numpy as np
import theano
import theano.tensor as T
import timeit

def ReLU(x):
    return max(x,0)
    
class EmbeddedAligment:
    """
    EmbeddedAlignment model specification in Theano.

    V, W - vectors of MNIST and spectrogram. Have shape n x d
    h - Embedded space dimension
    """

    def create_weight(self, row, col, name):
        return 
    
    def create_bias(self, row, name):
        return theano.shared(value=np.random.normal(size=(row,1)), name=name, borrow=True)
        
        
    def __init__(self, V, W,  h=512):
        
        (V_dim, W_dim) = (5,3) #(V.shape[0], W.shape[0]) # change this to suit
        
        # Model feature data. Set of data points.
        self.V = V
        self.W = W
        
        # Initialise parameters
        self.h = h
        self.W_m = theano.shared(value=np.random.normal(size=(h, V_dim)),name='W_m', borrow=True) #self.create_weight(row=h, col=V_dim, name='W_m')
        self.W_d = theano.shared(value=np.random.normal(size=(h, W_dim)),name='W_d', borrow=True)
        self.b_m = theano.shared(value=np.random.normal(size=(h,1)), name='b_m', borrow=True)    #self.create_bias(row=h, name='b_m')
        self.b_d = theano.shared(value=np.random.normal(size=(h,1)), name='b_d', borrow=True)
        
        print self.W_m
        # Initialize minibatch size
        self.batch_size = 40
        
        # Model parameters
        self.params = [self.W_m, self.b_m, self.W_d, self.b_d]

    ### SIMPLE HELPER FUNCTIONS ###
        
    def similarity(self, Y, X):
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
        n1, d1 = Y.shape
        n2, d2 = X.shape
        print (n1,d1), (n2,d2), 'AY'
        assert n1==n2 and d1==d2, 'embedded vector shapes do not match {0} {1} {2} {3}'.format(n1, n2, d1,d2)
        
        return ReLU(np.dot(Y,X))
    
    def embed_v(self):
        ''' Embeds a V matrix into h-dimensional space
            Each V matrix has dimension n x d, where d is 
            the dimension of the original data.
            
            The output Y has dimension n x h, where h is 
            the dimension of the embedded space.
            
        '''
#        n, d = self.V.shape
#        assert self.W_m.shape == (self.h,d), 'weight matrix for v has the wrong shape'
#        assert self.b_m.shape == (self.h,1), 'bias matrix for v has the wrong shape' 
        V_T = self.V.T # Now we have a d x n matrix
        
        Y_T = np.dot(self.W_m,V_T) + self.b_m.T
        Y =Y_T.T
#        assert Y.shape == (n,self.h), 'embedded version of v has the wrong shape'
        return Y
        
    def embed_w(self):
        ''' Embeds a W matrix into h-dimensional space
            Each W matrix has dimension n x d, where d is 
            the dimension of the original data.
            
            The output X has dimension n x h, where h is 
            the dimension of the embedded space.'''
#        n, d = self.W.shape
#        assert self.W_d.shape == (self.h,d), 'weight matrix for w has the wrong shape'
#        assert self.b_d.shape == (self.h,1), 'bias matrix for w has the wrong shape' 
        W_T = self.W.T # Now we have a d x n matrix
         
        X_T = np.dot(self.W_d,W_T) + self.b_d.T
        X =X_T.T
#        assert X.shape == (n,self.h), 'embedded version of x has the wrong shape'
        return T.nnet.relu(X)
#        return ReLU(X)        
    ################################
        
    def cost_function(self):
        ''' Given a matrix of v, w vectors computes the cost function given
            the current parameters
            
            '''
        Y = self.embed_v()
        X = self.embed_w()
        
        
        similarity_matrix = self.similarity(Y,X)
        print 'SIM:', similarity_matrix.eval()
        # [S_11 S_12 S_13]
        # [S_21 S_22 S_23]
        # [S_31 S_32 S_33]
        diag = T.nlinalg.ExtractDiag  
        print 'DIAG: ',diag.eval()  
        diag = diag.T
        # [S_11 S_22 S_33]
        
        # Subtract the diagonal to normalize each similarity score
        # [S_11-S_11 S_12-S_22 S__13-S_33]
        # [S_21-S_11 S_22-S_22 S_23-S_33]
        # [S_31-S_11 S_32-S_22 S_33-S_33]

        costs = similarity_matrix - diag
        print 'Costs:', costs.eval()
        
        return costs.elemwise.CAReduce(scalar_op='add')
        
        
#        #There may be a more numpy way to do this, but for now deal with 
#        #these for loops
#        cost = 0
#        for k in range(len(Y)):
#            for l in range(len(X)):
#                cost += ReLU(self.similarity(Y[k],X[l]) - self.similarity(Y[k],X[k]) + 1)
#        return cost
        
    def errors(self, v, w):
        ''' A pair v, w constitute an error if either the prediction from
        
        '''            
        ## TODO: IF WE WANT TO KEEP TRACK OF ERROR

def shared_dataset((x,y), borrow=True):
    """
    Create Theano shared variables to allow the data to
    be copied to the GPU to avoid performance-degrading
    sequential minibatch data copying.
    """
    shared_x = theano.shared(
        np.asarray(
            x, dtype=theano.config.floatX
        ), borrow=borrow
    )
    shared_y = theano.shared(
        np.asarray(
            y, dtype=theano.config.floatX
        ), borrow=borrow
    )
    return shared_x, shared_y
    
def stoch_grad_desc_train_model(datasets, gamma=1e-6, epochs=60, B=20, momentum=0.9):
    """
    Train the embedded alignment model using stochastic gradient descent.
    datasets = tuples of (W,V) for train, validate and test.
    
    gamma - Step size or "learning rate" for gradient descent
    epochs - Maximum number of epochs to run SGD
    momentum - TBH I dunno why we use this
    B - The batch size for each minibatch
    """

    # Obtain the correct dataset partitions
    train_set_w, train_set_v = shared_dataset(datasets[0])
    valid_set_w, valid_set_v = shared_dataset(datasets[1])
    test_set_w, test_set_v = shared_dataset(datasets[2])

    # Calculate the number of minibatches for each of
    # the training, validation and test dataset partitions

    n_train_batches = train_set_w.get_value(borrow=True).shape[0] // B
#    n_valid_batches = valid_set_w.get_value(borrow=True).shape[0] // B
#    n_test_batches = test_set_w.get_value(borrow=True).shape[0] // B

    # BUILD THE MODEL
    # ===============
    print("Building the logistic regression model...")

    # Create symbolic variables for the minibatch data
    index = T.lscalar()  # Integer scalar value
    V = T.vector('V')  # Feature vectors from MNIST
    W = T.vector('W')  # Feature vectors from TIDIGIT

    # Instantiate the logistic regression model and assign the cost
    embed = EmbeddedAligment(V=V, W=W, h=512)
    cost = embed.cost_function()  # This is what we minimize with SGD

#    # Compiles a set of Theano functions for both the testing
#    # and validation set that compute the errors on the model
#    # in a particular minibatch
#
#    test_model = theano.function(
#        inputs=[index],
#        outputs=embed.errors(v,w),
#        givens={v: test_set_v[index * B: (index + 1) * B],
#                w: test_set_w[index * B: (index + 1) * B]})
#
#    validate_model = theano.function(
#        inputs=[index],
#        outputs=embed.errors(v,w),
#        givens={v: valid_set_v[index * B: (index + 1) * B],
#                w: valid_set_w[index * B: (index + 1) * B]})
    
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
        (embed.W_m, embed.W_m - gamma * grad_W_m + (1. - momentum)*grad_W_m),
        (embed.b_m, embed.b_m - gamma * grad_b_m + (1. - momentum)*grad_b_m)
        (embed.W_d, embed.W_d - gamma * grad_W_d + (1. - momentum)*grad_W_d),
        (embed.b_d, embed.b_d - gamma * grad_b_d + (1. - momentum)*grad_b_d)
    ]

    # Similar to the above compiled Theano functions, except that 
    # it is carried out on the training data AND updates the parameters
    # W, b as it evaluates the cost for the particular minibatch
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            V: train_set_v[index * B: (index + 1) * B],
            W: train_set_w[index * B: (index + 1) * B]
        }
    )

    # TRAIN THE MODEL
    # ===============
    print("Training the embedded alignment model...")

    # Set parameters to stop minibatch early 
    # if performance is good enough
    patience = 5000  # Minimum number of examples to look at
    improvement_threshold = 0.995  # Relative improvement threshold   
    # Train through this number of minibatches before
    # checking performance on the validation set
    validation_frequency = min(n_train_batches, patience // 2)

    # Keep track of the validation loss and test scores
#    best_validation_loss = np.inf
#    test_score = 0.
    start_time = timeit.default_timer()

    # Begin the training loop
    # The outer while loop loops over the number of epochs
    # The inner for loop loops over the minibatches
    finished = False
    cur_epoch = 0
    prev_cost = np.inf
    while (cur_epoch < epochs) and (not finished):
        cur_epoch = cur_epoch + 1
        # Minibatch loop
        for minibatch_index in range(n_train_batches):
            # Calculate the average likelihood for the minibatches
            minibatch_avg_cost = train_model(minibatch_index)
            
            # Cut off training if no significant change in the loss function
            if abs(minibatch_avg_cost-prev_cost) < 1e-3:
                finished = True
                
            prev_cost = minibatch_avg_cost                 
            iteration = (cur_epoch - 1) * n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:
                print 'Loss function: ', minibatch_avg_cost
#                this_validation_loss = np.mean(validation_losses)
#
#                # Output current validation results
#                print(
#                    "Epoch %i, Minibatch %i/%i, Validation Error %f %%" % (
#                        cur_epoch,
#                        minibatch_index + 1,
#                        n_train_batches,
#                        this_validation_loss * 100.
#                    )
#                )
#
#                # If we obtain the best validation score to date
#                if this_validation_loss < best_validation_loss:
#                    # If the improvement in the loss is within the
#                    # improvement threshold, then increase the 
#                    # number of iterations ("patience") until the next check
#                    if this_validation_loss < best_validation_loss *  \
#                        improvement_threshold:
#                        patience = max(patience, iter * patience_increase)
#                    # Set the best loss to the new current (good) validation loss
#                    best_validation_loss = this_validation_loss
#
#                    # Now we calculate the losses on the minibatches 
#                    # in the test data set. Our "test_score" is the 
#                    # mean of these test losses
#                    test_losses = [
#                        test_model(i)
#                        for i in range(n_test_batches)
#                    ]
#                    test_score = np.mean(test_losses)
#
#                    # Output the current test results (indented)
#                    print(
#                        (
#                            "     Epoch %i, Minibatch %i/%i, Test error of"
#                            " best model %f %%"
#                        ) % (
#                            cur_epoch,
#                            minibatch_index + 1,
#                            n_train_batches,
#                            test_score * 100.
#                        )
#                    )
#
#                    # Serialise the model to disk using pickle
#                    with open('best_model.pkl', 'wb') as f:
#                        pickle.dump(logreg, f)
            # If the iterations exceed the current "patience"
            # then we are finished looping for this minibatch
#            if iteration > patience:
#                done_looping = True
#                break

    end_time = timeit.default_timer()
    # Serialise the model to disk using pickle
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(embed, f)
    print(
        (
            "Optimization complete with "
            "final loss: %f %%"
        ) % (minibatch_avg_cost)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec' % (
            cur_epoch, 
            1. * cur_epoch / (end_time - start_time)
        )
    )
    print("The code ran for %.1fs" % (end_time - start_time))


toy_data_train = (np.array([[1,2,3,4,5],[3,4,5,6,7]]), np.array([[1,2,3],[3,4,5]]))
toy_data_val = (np.array([[1,2,3,4,5],[3,4,5,6,7]]), np.array([[1,2,3],[3,4,5]]))
toy_data_test = (np.array([[1,2,3,4,5],[3,4,5,6,7]]), np.array([[1,2,3],[3,4,5]]))

datasets = [toy_data_train, toy_data_val, toy_data_test]
stoch_grad_desc_train_model(datasets, gamma=1e-6, epochs=60, B=20, momentum=0.9)