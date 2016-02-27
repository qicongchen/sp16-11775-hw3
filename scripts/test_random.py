#!/bin/python
import sys
import random

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: {0} output_file".format(sys.argv[0])
        print "output_file -- path to save the prediction score"
        exit(1)

    output_file = sys.argv[1]

    video_ids = []
    # read in labels
    label_file = "list/test"
    fread_label = open(label_file, 'r')
    for line in fread_label.readlines():
        tokens = line.strip().split(' ')
        video_id = tokens[0]
        video_ids.append(video_id)
    fread_label.close()

    # read in features
    scores = []
    for video_id in video_ids:
        score = random.uniform(-1, 1)
        scores.append(score)

    # dump result
    fwrite = open(output_file, 'w')
    for score in scores:
        fwrite.write("%s\n" % str(score))
    fwrite.close()
