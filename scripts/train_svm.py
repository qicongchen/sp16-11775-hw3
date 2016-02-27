#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print "Usage: {0} event_name feat_dir feat_suffix feat_type feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_suffix -- suffix of feature files, eg: spbof"
        print "feat_type -- type of feature files, dense|sparse"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_suffix = sys.argv[3]
    feat_type = sys.argv[4]
    feat_dim = int(sys.argv[5])
    output_file = sys.argv[6]

    video_ids = []
    # read in labels
    labels = []
    label_file = "list/"+event_name+"_train"
    fread_label = open(label_file, 'r')
    for line in fread_label.readlines():
        tokens = line.strip().split(' ')
        video_id = tokens[0]
        if tokens[1] == "NULL":
            label = -1
        else:
            label = 1
        video_ids.append(video_id)
        labels.append(label)
    fread_label.close()

    # read in features
    features = []
    for video_id in video_ids:
        feat_path = feat_dir + video_id + "." + feat_suffix
        feature = [0]*feat_dim
        if os.path.exists(feat_path) is True:
            if feat_type == 'dense':
                feature = numpy.genfromtxt(feat_path, delimiter=';')
            else:
                line = numpy.genfromtxt(feat_path, delimiter=' ', dtype=str)
                for item in line:
                    if len(item) == 0:
                        continue
                    tokens = item.split(':')
                    key = int(tokens[0])-1
                    value = float(tokens[1])
                    feature[key] = value
        features.append(feature)

    # train svm
    clf = SVC(probability=True)
    clf.fit(features, labels)
    # Dump model
    with open(output_file, 'wb') as f:
        cPickle.dump(clf, f)

    print 'SVM trained successfully for event %s!' % (event_name)
