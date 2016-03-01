#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 9:
        print "Usage: {0} event_name validation_part model_file feat_dir feat_suffix feat_type feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "validation_part -- part index of the validation file"
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_suffix -- suffix of feature files, eg: spbof"
        print "feat_type -- type of feature files, dense|sparse"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the prediction score"
        exit(1)

    event_name = sys.argv[1]
    validation_part = sys.argv[2]
    validation_file = "list/train_dev_part"+validation_part
    model_file = sys.argv[3]
    feat_dir = sys.argv[4]
    feat_suffix = sys.argv[5]
    feat_type = sys.argv[6]
    feat_dim = int(sys.argv[7])
    output_file = sys.argv[8]

    # load the kmeans model
    svm = cPickle.load(open(model_file, "rb"))

    video_ids = []
    # read in labels
    if validation_part != "-1":
        label_file = validation_file
        fread_label = open(label_file, 'r')
        fwrite = open("list/"+event_name+"_part"+validation_part+"_test_label", 'w')
        for line in fread_label.readlines():
            tokens = line.strip().split(' ')
            video_id = tokens[0]
            if tokens[1] != event_name:
                label = 0
            else:
                label = 1
            fwrite.write("%d\n" % label)
            video_ids.append(video_id)
        fwrite.close()
        fread_label.close()
    else:
        label_file = "list/test.video"
        fread_label = open(label_file, 'r')
        for line in fread_label.readlines():
            tokens = line.strip().split(' ')
            video_id = tokens[0]
            video_ids.append(video_id)
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

    # test svm
    scores = [sample[1] for sample in svm.predict_log_proba(features)]
    # dump result
    fwrite = open(output_file, 'w')
    for score in scores:
        fwrite.write("%s\n" % str(score))
    fwrite.close()
