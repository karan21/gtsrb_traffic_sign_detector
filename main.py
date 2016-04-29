#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from datasets import gtsrb
from sklearn.neighbors import KNeighborsClassifier

def main():
    #features = [None, 'gray', 'rgb', 'hsv', 'surf', 'hog']
    features = [ 'hog','gray',None]
    for f in xrange(len(features)):
        print "feature", features[f]
        (X_train, y_train), (X_test, y_test) = gtsrb.load_data(
            "datasets/gtsrb_training",
            feature=features[f],
            test_split=0.2,
            seed=42)

        # convert to numpy
        X_train = np.squeeze(np.array(X_train)).astype(np.float32)
        y_train = np.array(y_train)
        X_test = np.squeeze(np.array(X_test)).astype(np.float32)
        y_test = np.array(y_test)
        # find all class labels
        labels = np.unique(np.hstack((y_train, y_test)))    
        print '---knn---'
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        result = neigh.predict(X_test) 
        # knn = cv2.KNearest()
        # knn.train(X_train, y_train)
        # ret,result,neighbours,dist = knn.find_nearest(X_test,k=3)
        # print len(X_test), result.size
        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == result[i]:
                correct = correct + 1
        print correct
        accuracy = correct*100.0/result.size
        print accuracy

        print '----svm-------'
     #    clf = svm.LinearSVC()
     #    clf.fit(X_train, y_train)
     #    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     # intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     # multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     # verbose=0)
     #    result = clf.predict(X_test)
        print 'one vs rest'
        result = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
        # result = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == result[i]:
                correct = correct + 1
        print correct
        accuracy = correct*100.0/result.size
        print accuracy
        print '---one vs one'
        result = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == result[i]:
                correct = correct + 1
        print correct
        accuracy = correct*100.0/result.size
        print accuracy


if __name__ == '__main__':
    main()
