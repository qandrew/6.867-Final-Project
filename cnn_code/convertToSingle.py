# Process Data
# Given large batch of data, convert them to single .txt files based on the number
# Andrew Xia
# Dec 9 2016
from __future__ import print_function
import numpy as np

#first do MNIST
# activations = np.loadtxt(open("featureVector/mnistLastLayer512.txt"))
# print activations.shape
# values = np.loadtxt(open("featureVector/mnistYVal512.txt"))
# print values.shape
# print values[:10]

# output = [[] for i in xrange(10)]

# for i in xrange(len(activations)):
# 	# print int(values[i])
# 	output[int(values[i])].append(activations[i])

# print len(output[1])
# print output[1][0][300:310]

# for i in xrange(10):
# 	np.savetxt('mnistAct/'+str(i)+'.txt',output[i])

# print "done MNIST"

#now do spectro 1024
activations = np.loadtxt(open("featureVector/cnnLastLayerVal1024.txt"))
print(activations.shape)
values = np.loadtxt(open("featureVector/cnnActualYVal1024.txt"))
print(values.shape)
# print(values[:10])
for i in xrange(10):
	print(np.argmax(values[i]),end=" ")
print(" ")

output = [[] for i in xrange(11)]

for i in xrange(len(activations)):
	loc = np.argmax(values[i])
	output[loc].append(activations[i])

# print len(outout)
print(len(output[1]))
print(len(output[1][0]))
# print(output[1][0][300:310])

for i in xrange(1,10):
	np.savetxt('spectro1024Act/'+str(i)+'.txt',output[i])
np.savetxt('spectro1024Act/z.txt',output[0]) #manually rename
np.savetxt('spectro1024Act/o.txt',output[10]) #manually rename

print("done spectro 1024")

#now do spectro 512
activations = np.loadtxt(open("featureVector/cnnLastLayerVal512.txt"))
print(activations.shape)
values = np.loadtxt(open("featureVector/cnnActualYVal512.txt"))
print(values.shape)
# print(values[:10])
for i in xrange(10):
	print(np.argmax(values[i]),end=" ")
print(" ")

output = [[] for i in xrange(11)]

for i in xrange(len(activations)):
	loc = np.argmax(values[i])
	output[loc].append(activations[i])

# print len(outout)
print(len(output[1]))
print(len(output[1][0]))
# print(output[1][0][300:310])

for i in xrange(1,10):
	np.savetxt('spectro512Act/'+str(i)+'.txt',output[i])
np.savetxt('spectro512Act/z.txt',output[0]) #manually rename
np.savetxt('spectro512Act/o.txt',output[10]) #manually rename

print("done spectro 512")