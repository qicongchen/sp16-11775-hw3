#!/bin/python 

import numpy
import sys


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} event_name validation_part output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "validation_part -- part index of the validation file"
        print "output_file -- path to save the prediction score"
        exit(1)

    event_name = sys.argv[1]
    validation_part = sys.argv[2]
    if validation_part != "-1":
        pred_file_suffix = event_name+"_part"+validation_part+"_pred"
    else:
        pred_file_suffix = event_name+"_pred"
    pred_files = ["mfcc_pred/"+pred_file_suffix, "cnn_pred/"+pred_file_suffix]
    output_file = sys.argv[3]

    # late fusion
    scores = []
    for pred_file in pred_files:
        vector = numpy.genfromtxt(pred_file)
        if len(scores) == 0:
            scores = [0]*len(vector)
        scores = numpy.add(scores, vector)
    scores = scores/len(pred_files)
    numpy.savetxt(output_file, scores)
