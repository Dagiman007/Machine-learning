"""
@author: Dagmawi Abraham Seifu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from itertools import groupby


#define the euclidean and the manhattan distance between two data points for breast cancer dataset
def Euclidean_distance(x1,x2, i,j):
    distance = np.sum(np.square(x1.loc[i].iloc[1:len(x1.columns)] - x2.loc[j].iloc[1:len(x2.columns)]))
    return np.sqrt(distance)
                           

def knn(training_set, test_set, k):
    result = {}
    for j in test_set.index:
        bucket = {}
        for i in training_set.index:
            dist = Euclidean_distance(training_set, test_set, i, j)
            value = training_set.loc[i].iloc[0]
            bucket[dist] = value

        k_pairs = {n:bucket[n] for n in sorted(bucket.keys())[:k]}
        vote = {value:len(list(freq)) for value, freq in groupby(sorted(k_pairs.values()))}
        result[j] = sorted(vote.items(),key = operator.itemgetter(1),reverse=True)[0][0]

    return result

def getAccuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set.values[x][0] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def main():
    #load the training sets and the test sets
    names = ['sample-code','thickness','cell-size','cell-shape','marginal-adhesion',
             'epithelial-cell-size','bare-nuclei','bland-chromatin','normal-nucleoli','mitoses','class']
    X_train = pd.read_csv('Data//Breast Cancer//training_big.csv', names=names)
    X_test = pd.read_csv('Data//Breast Cancer//test_small.csv', names=names)
    Y_train = X_train.iloc[:,10:]
    Y_test = X_test.iloc[:,10:]
    
    errors = []
    accuracy = {}
    for k in range(5,6,2):
        result = knn(X_train,X_test,k)
        #accuracy[k] = getAccuracy(Y_test,result)
        accuracy =round(sum([1 for j in X_test.index if X_test.loc[j].iloc[0] == result[j] ])/float(len(X_test))*100,2)
    
    

main()


