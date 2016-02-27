#!/bin/python
# Randomly select 

import numpy
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of video names"
        print "select_ratio -- the ratio of frames to be randomly selected from each audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]
    output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list, "r")
    fwrite = open(output_file, "w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        sift_dir = "sift/" + video_id + "/"
        if os.path.exists(sift_dir) is False:
            continue
        for sift_file in os.listdir(sift_dir):
            if '.sift' not in sift_file:
                continue
            # if no key point, why an empty file??
            if os.stat(sift_dir+sift_file).st_size == 0:
                os.remove(sift_dir+sift_file)
                continue
            sift_id = sift_file.split('.sift')[0]

            array = numpy.genfromtxt(sift_dir+sift_file, delimiter=";")
            # if only one key point
            if len(array.shape) == 1:
                continue
            numpy.random.shuffle(array)
            select_size = int(array.shape[0] * ratio)
            feat_dim = array.shape[1]

            for n in xrange(select_size):
                line = str(array[n][0])
                for m in range(1, feat_dim):
                    line += ';' + str(array[n][m])
                fwrite.write(line + '\n')
    fwrite.close()
