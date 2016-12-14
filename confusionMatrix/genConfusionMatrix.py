# generate data for a confusion matrix
# Andrew Xia
# Dec 13 2016
# see mnist_to_spec, spec_to_mnist rtf files for data

def minstToSpec():
	#define 10 as z, 0 as o

	#mnist to spec
	actualTotal = []
	outputTotal = []

	#generate actual labels
	for i in xrange(10):
		actualTotal.extend([i for j in xrange(100)])

	#0 case
	output = [0 for i in xrange(95)]
	output.extend([7,6,7,2,7])
	outputTotal.extend(output)

	#1 case
	output = [1 for i in xrange(74)]
	output.extend([7 for i in xrange(9)])
	output.extend([4 for i in xrange(8)])
	output.extend([9,6])
	output.extend([5 for i in xrange(5)])
	output.extend([3 for i in xrange(2)])
	outputTotal.extend(output)

	#2 case
	output = [2 for i in xrange(80)]
	output.extend([3 for i in xrange(6)])
	output.extend([5 for i in xrange(2)])
	output.extend([7 for i in xrange(6)])
	output.extend([6,8,8,1,0,0]) #one 0 is a 10
	outputTotal.extend(output)

	#3 case
	output = [3 for i in xrange(94)]
	output.extend([5,2,8,7,8,7]) #define 10 as 0
	outputTotal.extend(output)

	#4 case
	output = [4 for i in xrange(92)]
	output.extend([0,5,0,2,6,9,0,5]) #one 0 is a 10
	outputTotal.extend(output)

	#5 case
	output = [5 for i in xrange(98)]
	output.extend([9,1]) 
	outputTotal.extend(output)

	#6 case
	output = [6 for i in xrange(96)]
	output.extend([4,0,0,5]) #two 0 is a 10
	outputTotal.extend(output)

	#7 case
	output = [7 for i in xrange(97)]
	output.extend([5,5,0]) 
	outputTotal.extend(output)

	#8 case
	output = [8 for i in xrange(88)]
	output.extend([5 for i in xrange(4)])
	output.extend([3 for i in xrange(5)])
	output.extend([1,6,2]) 
	outputTotal.extend(output)

	#9 case
	output = [9 for i in xrange(71)]
	output.extend([7 for i in xrange(8)])
	output.extend([5 for i in xrange(16)])
	output.extend([3,6,6,4,8]) 
	outputTotal.extend(output)

	#labels
	labels = [i for i in xrange(10)]

	print len(actualTotal)
	print len(outputTotal)
	return outputTotal,actualTotal, labels


def specToMnist():

	#mnist to spec
	actualTotal = []
	outputTotal = []

	#generate actual labels
	for i in xrange(10):
		actualTotal.extend([i for j in xrange(100)])
	# actualTotal.extend([0 for j in xrange(100)])
	actualTotal.extend(['z' for j in xrange(100)])

	#0 case
	output = [0 for i in xrange(66)]
	output.extend([5 for i in xrange(4)])
	output.extend([4 for i in xrange(14)])
	output.extend([7 for i in xrange(5)])
	output.extend([1 for i in xrange(4)])
	output.extend([6 for i in xrange(4)])
	output.extend([8,8,8])
	outputTotal.extend(output)

	#1 case
	output = [1 for i in xrange(97)]
	output.extend([4,8,4])
	outputTotal.extend(output)

	#2 case
	output = [2 for i in xrange(82)]
	output.extend([8 for i in xrange(8)])
	output.extend([0 for i in xrange(5)])
	output.extend([6,6,7,4,4]) 
	outputTotal.extend(output)

	#3 case
	output = [3 for i in xrange(82)]
	output.extend([8 for i in xrange(11)])
	output.extend([4,7,7,7,5,0,6]) #define 10 as 0
	outputTotal.extend(output)

	#4 case
	output = [4 for i in xrange(100)]
	outputTotal.extend(output)

	#5 case
	output = [5 for i in xrange(89)]
	output.extend([7,7,7,6,6,6,3,0,4,8,9]) 
	outputTotal.extend(output)

	#6 case
	output = [6 for i in xrange(98)]
	output.extend([4,8]) #two 0 is a 10
	outputTotal.extend(output)

	#7 case
	output = [7 for i in xrange(91)]
	output.extend([8,6,4,5,8,1,8,4,6]) 
	outputTotal.extend(output)

	#8 case
	output = [8 for i in xrange(96)]
	output.extend([6,0,3,6]) 
	outputTotal.extend(output)

	#9 case
	output = [9 for i in xrange(79)]
	output.extend([4 for i in xrange(5)])
	output.extend([1 for i in xrange(6)])
	output.extend([5,5,0,0,8,8,3,7,7,7]) 
	outputTotal.extend(output)

	#z case
	output = ['z' for i in xrange(84)]
	output.extend([1 for i in xrange(6)])
	output.extend([6 for i in xrange(5)])
	output.extend([4,4,8,9,7])
	outputTotal.extend(output)

	#labels
	labels = [i for i in xrange(10)]
	labels.extend(['z'])

	print len(actualTotal)
	print len(outputTotal)
	return outputTotal,actualTotal, labels


specToMnist()