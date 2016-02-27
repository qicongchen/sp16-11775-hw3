#!/bin/python
import numpy
import os
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: {0} file_list".format(sys.argv[0])
        print "file_list -- the list of videos"
        exit(1)

    file_list = sys.argv[1]

    fread = open(file_list, "r")

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        mfcc_path = "mfcc/" + video_id + ".mfcc.csv"
        if os.path.exists(mfcc_path) is False:
            continue
        X = numpy.genfromtxt(mfcc_path, delimiter=";")
        vector = numpy.average(X, axis=0)
        s = numpy.sum(vector) + 0.0
        if s > 0:
            vector = vector/s
        line = ';'.join([str(v) for v in vector])
        feat_path = "raw/"+video_id+".feat"
        fwrite = open(feat_path, "w")
        fwrite.write(line + '\n')
        fwrite.close()
    fread.close()

    print "Raw features generated successfully!"
