#!/usr/bin/env python
import cv2
import numpy as np
import hog
import csv
from matplotlib import cm
from matplotlib import pyplot as plt

def load_data(rootpath="datasets/gtsrb_training", feature=None, cut_roi=True,
              test_split=0.2, plot_samples=False, seed=113):
    
    classes = np.arange(0, 43, 1)

    # read all training samples and corresponding class labels
    X = []  # data
    labels = []  # corresponding labels
    print len(classes)
    for c in xrange(len(classes)):
        # subdirectory for class
        prefix = rootpath + '/' + format(classes[c], '05d') + '/'
        #print prefix
        # annotations file
        gt_file = open(prefix + 'GT-' + format(classes[c], '05d') + '.csv')

        # csv parser for annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')
        gt_reader.next()  # skip header
        # loop over all images in current annotations file
        i = 0
        for row in gt_reader:
            # print row
            # print i
            i = i + 1
            # first column is filename
            im = cv2.imread(prefix + row[0])
            # remove regions surrounding the actual traffic sign
            if cut_roi:
                im = im[np.int(row[4]):np.int(row[6]),
                        np.int(row[3]):np.int(row[5]), :]

            X.append(im)
            labels.append(c)
        gt_file.close()

    # perform feature extraction
    X = _extract_feature(X, feature)
    X_train = X
    y_train = labels

    X_test = []  # data
    y_test = []  # corresponding labels
    print 'Testing'
    gt_file = open('datasets/gtsrb_training/final_test/GT-final_test.csv')
    prefix = 'datasets/gtsrb_training/final_test/'
    # csv parser for annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')
    gt_reader.next()  # skip header
    # loop over all images in current annotations file
    for row in gt_reader:
        # first column is filename
        im = cv2.imread(prefix + row[0])
        # remove regions surrounding the actual traffic sign
        if cut_roi:
            im = im[np.int(row[4]):np.int(row[6]),
                    np.int(row[3]):np.int(row[5]), :]

        X_test.append(im)
        y_test.append(np.int(row[7]))
    gt_file.close()

    # perform feature extraction
    X_test = _extract_feature(X_test, feature)
    print len(X_test), len(y_test), len(X_train), len(y_train)
    return (X_train, y_train), (X_test, y_test)


def _extract_feature(X, feature):

    # transform color space
    if feature == 'gray' or feature == "surf":
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]

    # operate on smaller image
    small_size = (32, 32)
    X = [cv2.resize(x, small_size) for x in X]

    # extract features

    
    if feature == 'hog':
        # histogram of gradients
        block_size = (small_size[0] / 2, small_size[1] / 2)
        block_stride = (small_size[0] / 4, small_size[1] / 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(small_size, block_size, block_stride,
                                cell_size, num_bins)
        # X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
        # X = [hog.hog(image=x,pixels_per_cell=block_size,cells_per_block=cell_size) for x in X]
    
    X = [x.flatten() for x in X]
    return X
