#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print "Usage: {0} event_name validation_part feat_dir feat_suffix feat_type feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "validation_part -- part index of the validation file"
        print "feat_dir -- dir of feature files"
        print "feat_suffix -- suffix of feature files, eg: spbof"
        print "feat_type -- type of feature files, dense|sparse"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    validation_part = sys.argv[2]
    validation_file = "list/train_dev_part"+validation_part
    feat_dir = sys.argv[3]
    feat_suffix = sys.argv[4]
    feat_type = sys.argv[5]
    feat_dim = int(sys.argv[6])
    output_file = sys.argv[7]
    all_file = "list/train_dev"

    validation_video_ids = []
    # read in labels
    if validation_part != "-1":
        label_file = validation_file
        fread_label = open(label_file, 'r')
        for line in fread_label.readlines():
            tokens = line.strip().split(' ')
            video_id = tokens[0]
            validation_video_ids.append(video_id)
        fread_label.close()

    video_ids = []
    # read in labels
    labels = []
    label_file = all_file
    fread_label = open(label_file, 'r')
    for line in fread_label.readlines():
        tokens = line.strip().split(' ')
        video_id = tokens[0]
        if video_id in validation_video_ids:
            continue
        if tokens[1] != event_name:
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
                if len(line.shape) == 0:
                    line = numpy.array([line])
                for item in line:
                    if len(item) == 0:
                        continue
                    tokens = item.split(':')
                    key = int(tokens[0])-1
                    value = float(tokens[1])
                    if key < feat_dim:
                        feature[key] = value
        features.append(feature)

    # train svm
    clf = SVC(probability=True)
    clf.fit(features, labels)
    # Dump model
    with open(output_file, 'wb') as f:
        cPickle.dump(clf, f)

    print 'SVM trained successfully for event %s!' % (event_name)
