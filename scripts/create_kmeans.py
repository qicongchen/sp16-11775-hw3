#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import collections
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, "rb"))

    fread = open(file_list, "r")

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        sift_dir = "sift/" + video_id + "/"
        if os.path.exists(sift_dir) is False:
            continue
        vector = [0]*cluster_num
        for sift_file in os.listdir(sift_dir):
            if '.sift' not in sift_file:
                continue
            # if no key point, why an empty file??
            if os.stat(sift_dir+sift_file).st_size == 0:
                os.remove(sift_dir+sift_file)
                continue
            sift_id = sift_file.split('.sift')[0]

            X = numpy.genfromtxt(sift_dir+sift_file, delimiter=";")
            # if only one key point
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            labels = kmeans.predict(X)
            counter = collections.Counter(labels)
            frame_vector = [counter[n] for n in xrange(cluster_num)]
            vector = numpy.add(vector, frame_vector)
        s = numpy.sum(vector) + 0.0
        if s > 0:
            vector = vector/s
        line = ';'.join([str(v) for v in vector])
        feat_path = "kmeans/"+video_id+".feat"
        fwrite = open(feat_path, "w")
        fwrite.write(line + '\n')
        fwrite.close()
    fread.close()

    print "K-means features generated successfully!"
