# generate a confusion matrix
# Andrew Xia
# Dec 13 2016

print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import genConfusionMatrix #data

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


mnist = genConfusionMatrix.minstToSpec()
matr = confusion_matrix(mnist[1],mnist[0]) #test
plt.figure()
plot_confusion_matrix(matr, classes=mnist[2],
                      title='Image Annotation Confusion Matrix')
plt.savefig('mnistToSpec.png')

mnist = genConfusionMatrix.specToMnist()
matr = confusion_matrix(mnist[1],mnist[0]) #test
plt.figure()
plot_confusion_matrix(matr, classes=mnist[2],
                      title='Image Retrieval Confusion Matrix')
plt.savefig('specToMnist.png')


plt.show()