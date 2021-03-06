# -*- coding: utf-8 -*-
"""
This script is intended to convert all the spectogram data within subfolders to 
a numpy array of dimension n x d, where d is the dimension of the vector for a 
single spectogram, and n is the number of training points.

It will also classify each spectogram as 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'oh', 
based on the classification in the file name, assuming a file name of the format
'xx_<NUM>x.txt' so that <NUM> is one of the aforementioned values.

Other values of <NUM> will be discarded.

It will also create a corresponding Y value in the set R^11, again corresponding
to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'oh',   as aforementioned, in that order.

"""


import os
import numpy as np
import matplotlib.pyplot as plt


# Modify if running on your own computer
rootdir = '/home/sitara/test_single/test' #Sitara
#rootdir = '/home/andrew/Dropbox (MIT)/6867_Project/single_utterances/test/test1' #Andrew's code

bin_freq = 23 
spect_width = bin_freq  # Don't add one pixel of zeros on either side of the image
window_size = 100   
                    #Use a fixed window size, e.g. 100 frames and then paste the
                    #spectrogram of the digit in the middle. If the duration of the digit
                    #is shorter than 100 frames, then you're basically zero-padding the
                    #left and right of the spectrogram. If it's longer than 100 frames,
                    #then you chop off the excess. You don't have to use 100 frames, and
                    #could instead use 80 or 200 or something to accommodate the majority
                    #of the training examples. This allows your network to have a
                    #consistent input size for every training example (similar to how MNIST
                    #digits are all 28x28).
dim_Y = 11  
                    #digit recognition over the 11 classes in TIDIGITS (0 through
                    #9 plus "oh"),


def load_from_file(f):
    '''Given a file, returns a list of the string values in that value'''
    data = []
    for line in f:
        vector = []
        line = line.replace("[", "")
        line = line.replace("]", "")
        line_chars = line.split()
        for char in line_chars:
            vector.append(float(char))
        try:
            assert len(vector) == bin_freq
            data.append(vector)
        except AssertionError:
            if len(vector) == 0:
                pass
            else:
                print len(vector)
                raise AssertionError
        
    # Now we have a list of length-23 vectors which we need to trim/pad to 
    # window_size
    if len(data)>window_size:
        #cut excess rows
        cut = 1.*(len(data) - window_size)
        data = data[int(np.floor(cut/2)):-int(np.ceil(cut/2))]
    else:
        # pad data with excess rows of zeros about center
        cut = 1.*(window_size - len(data))
        data = [[0]*bin_freq]*int(np.floor(cut/2)) + data + [[0]*bin_freq]*int(np.ceil(cut/2))  
    #Convert data to a numpy array and invert it    
    data = np.flipud(np.array(data,dtype=np.float32))
    #Pad one pixel of zeros on top and bottom of array
    zero = np.zeros((bin_freq,))
    data[0] = zero
    data[-1] = zero
    return data
    
def get_data(rootdir):
    '''Given a directory, load all the files within it as described on top'''
    X = []
    Y = []
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            try:
                y = int(filename[3])
                y_val = np.zeros((dim_Y,1), dtype=np.float32)
                y_val[y] = 1
                f = open(os.path.join(subdir, filename))
                row = load_from_file(f)
                f.close()
                #check to ensure data has the right dimension
                assert (window_size, spect_width) == row.shape
                X.append(row)
                Y.append(y_val)
            except ValueError:
                if filename[3]=='o':
                    y_val = np.zeros((dim_Y,1))
                    y_val[dim_Y-1] = 1 #the last entry (index 10) set as 1 for 'o'
                    f = open(os.path.join(subdir, filename))
                    row = load_from_file(f)
                    f.close()
                    #check to ensure data has the right dimension
                    assert (window_size, spect_width) == row.shape
                    X.append(row)
                    Y.append(y_val)
                if filename[3]=='z':
                    y_val = np.zeros((dim_Y,1))
                    y_val[0] = 1 #the first entry (index 0) set as 1 for 'z' or zero
                    f = open(os.path.join(subdir, filename))
                    row = load_from_file(f)
                    f.close()
                    #check to ensure data has the right dimension
                    assert (window_size, spect_width) == row.shape
                    X.append(row)
                    Y.append(y_val)
    return np.array(X,dtype=np.float32), np.array(Y, dtype=np.float32)
    

if __name__ == "__main__":

    x,y = get_data(rootdir)
    np.savetxt('test.txt', x[0])
    for i in range(len(y)):       
        y_val = np.nonzero(y[i])[0][0]
        x_norm = x[i]/np.max(x[i]) - 0.5
        plt.imshow(x_norm, interpolation='none',cmap='gray')
        plt.savefig(rootdir+'/spect'+str(y_val)+'/image_gray_norm_'+str(i)+'.eps')
        plt.imshow(x[i], interpolation='none',cmap='gray')
        plt.savefig(rootdir+'/spect'+str(y_val)+'/image_gray_'+str(i)+'.eps')
        plt.imshow(x[i], interpolation='none')
        plt.savefig(rootdir+'/spect'+str(y_val)+'/image_'+str(i)+'.eps')
        
        if i > 100:
            break
