#!/bin/bash

# An example script for feature extraction of Homework 1

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.

# Paths to different tools; 
opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
opencv_path=/home/ubuntu/tools/opencv-2.4.10
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

# Two additional variables
video_path=../video   # path to the directory containing all the videos. In this example setup, we are linking all the videos to "../video"
cluster_num=200        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
                      # You need to explore the best config by yourself.
mkdir -p frame video sift kmeans

# This part does feature extraction, it may take quite a while if you have a lot of videos. Totally 3 steps are taken:
# 1. Video pre-processing: You may use the provided ffmpeg tool to down-sample videos into images. 
#    A reasonable setting for this would be: Select the first 30 seconds of the videos, 
#    downsize them to 160×120 pixels, and export 15 frames per second.
# 2. Keyframe selection: You may use ffmpeg to do the keyframe selection or implement other
#    keyframe pooling algorithms such as color histogram comparison to pool the keyframes from the
#    image set of a video in step 1. 
# 3. ExtractSIFT: You may use the provided OpenCV to extract the SIFT features from the selected
#    keyframes. There are many examples available online such as 1 and 2.
cat list/train | awk '{print $1}' > list/train.video
cat list/test | awk '{print $1}' > list/test.video
cat list/train.video list/test.video > list/all.video
for line in $(cat "list/all.video"); do
    ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t 30 -r 15 -vf scale=160x120,setdar=dar=4/3 video/${line}.mp4
    mkdir -p frame/${line} sift/${line}
    ffmpeg -y -i video/${line}.mp4 -vsync 2 -vf select='eq(pict_type\,I)' -f image2 frame/${line}/%d.jpeg
done
python scripts/extract_sift.py list/all.video

echo "Pooling SIFTs (optional)"
python scripts/select_frames.py list/train.video 0.2 select.sift.csv || exit 1;

echo "Training the k-means model"
python scripts/train_kmeans.py select.sift.csv $cluster_num kmeans.${cluster_num}.model || exit 1;

echo "Creating k-means cluster vectors"
python scripts/create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;


# Great! We are done!
echo "SUCCESSFUL COMPLETION"
