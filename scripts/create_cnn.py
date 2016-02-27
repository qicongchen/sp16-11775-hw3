import numpy as np
import sys
# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/../med
sys.path.insert(0, caffe_root + 'python')
import os
import caffe


if __name__ == '__main__':
    # GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    batch_size = 50
    batch_video_ids = []
    net.blobs['data'].reshape(batch_size, 3, 227, 227)

    video_id2vector = {}
    image_list = np.genfromtxt("image_list", dtype=str)
    for i, image_file in enumerate(image_list):
        if os.path.exists(image_file) is False:
            continue
        video_id = image_file.split('/')[1]
        if video_id not in video_id2vector:
            video_id2vector[video_id] = np.array([0]*4096)
        image = caffe.io.load_image(image_file)
        net.blobs['data'].data[len(batch_video_ids)] = transformer.preprocess('data', image)
        batch_video_ids.append(video_id)
        if len(batch_video_ids) < batch_size:
            continue
        out = net.forward()
        for j, video_id in enumerate(batch_video_ids):
            vector = net.blobs['fc7'].data[j]
            video_id2vector[video_id] = np.add(video_id2vector[video_id], vector)
        batch_video_ids = []
        print i

    # last batch
    if len(batch_video_ids) > 0:
        net.blobs['data'].reshape(len(batch_video_ids), 3, 227, 227)
        out = net.forward()
        for i, video_id in enumerate(batch_video_ids):
            vector = net.blobs['fc7'].data[i]
            video_id2vector[video_id] = np.add(video_id2vector[video_id], vector)
    # dump feature
    for video_id, vector in video_id2vector.items():
        s = np.sum(vector) + 0.0
        if s > 0:
            vector = vector/s
        line = ';'.join([str(v) for v in vector])
        feat_path = "cnn/"+video_id+".feat"
        fwrite = open(feat_path, "w")
        fwrite.write(line + '\n')
        fwrite.close()
    print("CNN features generated successfully!")
