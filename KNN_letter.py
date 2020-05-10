"""
@author: Dagmawi Abraham Seifu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator


#define the euclidean and the manhattan distance between two data points
def Euclidean_distance(x1,x2, length):
    distance = 0
    for x in range(1,length):
        distance += np.square(x1[x] - x2[x])
    
    return np.sqrt(distance)

def Manhattan_distance(x1,x2, length):
    distance = 0
    for x in range(length):
        distance += np.abs(x1[x] - x2[x])

    return distance
                           

def knn(training_set, test_instance, k):
    distances = {}
    length = test_instance.shape[0]
    for x in range(len(training_set)):
        dist = Euclidean_distance(test_instance.values, training_set.iloc[x].values, length)
        distances[x] = dist[0]
        sortdist = sorted(distances.items(), key = operator.itemgetter(1))
        
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x][0])
        
    count = {} # most frequent class of rows
    for x in range(len(neighbors)):
        response = training_set.iloc[neighbors[x]][-1]
        if response in count:
            count[response] += 1
        else:
            count[response] = 1
            
    sortcount = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
    return (sortcount[0][0], neighbors)

def getAccuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def main():
    #load the training sets and the test sets
    train_set = pd.read_csv('Data//Letter recognition//training_big.csv', header = None)
    test_set = pd.read_csv('Data//Letter recognition//test_small.csv', header = None)
    X_train = train_set.iloc[:,0:16]
    Ytrain = train_set.iloc[:,0]
    X_test = test_set.iloc[:,0:16]
    Y_test = test_set.iloc[:,0]

    # training and test dataset 2 with 60/40 proportion
    train_set2 = pd.read_csv('Data//Letter recognition//training_small.csv', header = None)
    test_set2 = pd.read_csv('Data//Letter recognition//test_big.csv', header = None)
    X_train2 = train_set2.iloc[:,0:16]
    Ytrain2 = train_set2.iloc[:,0]
    X_test2 = test_set2.iloc[:,0:16]
    Y_test2 = test_set2.iloc[:,0]

    errors = []
    accuracy = {}
    predictions = []
    print("First train\\test set\n")
    for k in range(1,10):
        for instance in range(test_set.shape[0]):
            result,neighbor = knn(train_set.iloc[:,1:10],test_set.iloc[instance],k)
            predictions.append(result)
        accuracy[k] = getAccuracy(test_set,predictions)
        print("k = " + k + ", Accuracy = " + accuracy[k])
    
    print('\n Train\\Test set 2\n')
    predictions2 = []
    accuracy2 = {}
    for k in range(1,10):
        for instance in range(test_set2.shape[0]):
            result2,neighbor2 = knn(train_set2.iloc[:,1:10], test_set2.iloc[instance],k)
            predictions2.append(result2)
        accuracy2[k] = getAccuracy(test_set2,predictions)
        print("k = " + k + ", Accuracy = " + accuracy2[k])
    

main()


