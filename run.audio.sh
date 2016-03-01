#!/bin/bash

# Paths to different tools; 
opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

# Two additional variables
video_path=../video   # path to the directory containing all the videos. In this example setup, we are linking all the videos to "../video"
cluster_num=200        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
                      # You need to explore the best config by yourself.
mkdir -p temp audio mfcc kmeans

# This part does feature extraction, it may take quite a while if you have a lot of videos. Totally 3 steps are taken:
# 1. ffmpeg extracts the audio track from each video file into a wav file
# 2. The wav file may contain 2 channels. We always extract the 1st channel using ch_wave
# 3. SMILExtract generates the MFCC features for each wav file
#    The config file MFCC12_0_D_A.conf generates 13-dim MFCCs at each frame, together with the 1st and 2nd deltas. So you 
#    will see each frame totally has 39 dims. 
#    Refer to Section 2.5 of this document http://web.stanford.edu/class/cs224s/hw/openSMILE_manual.pdf for better configuration
#    (e.g., normalization) and other feature types (e.g., PLPs )     
cat list/train_dev | awk '{print $1}' > list/train_dev.video
cat list/train_dev.video list/test.video > list/all.video
for line in $(cat "list/all.video"); do
    ffmpeg -y -i $video_path/${line}.mp4 -f wav temp/tmp.wav
    #ffmpeg -y -i $video_path/${line}.mp4 -ac 1 -f wav audio/$line.wav
    ch_wave temp/tmp.wav -c 0 -o audio/$line.wav
    SMILExtract -C config/MFCC12_E_D_A_Z.conf -I audio/$line.wav -O mfcc/$line.mfcc.csv
done
# You may find the number of MFCC files mfcc/*.mfcc.csv is slightly less than the number of the videos. This is because some of the videos
# don't hae the audio track. For example, HVC1221, HVC1222, HVC1261, HVC1794 

# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video. 
echo "Pooling MFCCs (optional)"
python scripts/select_frames.py list/train_dev.video 0.01 select.mfcc.csv || exit 1;

# now trains a k-means model using the sklearn package
echo "Training the k-means model"
python scripts/train_kmeans.py select.mfcc.csv $cluster_num kmeans.${cluster_num}.model || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 
echo "Creating k-means cluster vectors"
python scripts/create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;

# Now you can see that you get the bag-of-word representations under kmeans/. Each video is now represented
# by a {cluster_num}-dimensional vector.

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
