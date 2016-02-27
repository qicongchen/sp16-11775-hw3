#!/bin/python
import numpy
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    vocab_file = sys.argv[1]
    file_list = sys.argv[2]

    # create reverted index for vocab
    vocab = numpy.genfromtxt(vocab_file, dtype=str)
    vocab_index = {}
    n_doc = 0
    vocab_df = [0]*len(vocab)
    for i, v in enumerate(vocab):
        vocab_index[v] = i

    vectors = {}
    fread = open(file_list, "r")
    for line in fread.readlines():
        video_id = line.replace('\n', '')
        asr_path = "asr/" + video_id + ".ctm"
        if os.path.exists(asr_path) is False:
            continue
        n_doc += 1
        # word count
        vector = {}
        fread_asr = open(asr_path, "r")
        for line_asr in fread_asr.readlines():
            tokens = line_asr.strip().split(' ')
            word = tokens[4]
            if word not in vocab_index:
                continue
            if word not in vector:
                vector[word] = 0
            vector[word] += 1
        fread_asr.close()
        # inc df
        for k, v in vector.items():
            vocab_df[vocab_index[k]] += 1
        # word count dict to list
        tmp = [0]*len(vocab)
        for k, v in vector.items():
            tmp[vocab_index[k]] = v
        vector = tmp
        # save tf vector
        vectors[video_id] = vector

    vocab_idf = numpy.log(numpy.divide(n_doc, numpy.add(1.0, vocab_df)))
    for video_id, vector in vectors.items():
        # normalize by idf
        vector = vector * vocab_idf
        s = numpy.sum(vector) + 0.0
        if s > 0:
            vector = vector/s
        line = ';'.join([str(v) for v in vector])
        feat_path = "asrfeat/"+video_id+".feat"
        fwrite = open(feat_path, "w")
        fwrite.write(line + '\n')
        fwrite.close()
    fread.close()

    print "ASR features generated successfully!"
