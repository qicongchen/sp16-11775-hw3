#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
map_path=/home/ubuntu/tools/mAP
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$map_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

echo "#####################################"
echo "#       MED with Imtraj Features      #"
echo "#####################################"
mkdir -p imtraj_pred
# iterate over the events
feat_dim_imtraj=32768
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "imtraj/" "spbof" "sparse" $feat_dim_imtraj imtraj_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py imtraj_pred/svm.$event.model "imtraj/" "spbof" "sparse" $feat_dim_imtraj imtraj_pred/${event}_pred || exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_test_label imtraj_pred/${event}_pred
done

echo "#####################################"
echo "#       MED with Sift Features      #"
echo "#####################################"
mkdir -p sift_pred
# iterate over the events
feat_dim_sift=200
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "kmeans/" "feat" "dense" $feat_dim_sift sift_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py sift_pred/svm.$event.model "kmeans/" "feat" "dense" $feat_dim_sift sift_pred/${event}_pred || exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_test_label sift_pred/${event}_pred
done

echo "#####################################"
echo "#       MED with CNN Features      #"
echo "#####################################"
mkdir -p cnn_pred
# iterate over the events
feat_dim_cnn=4096
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py cnn_pred/svm.$event.model "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/${event}_pred || exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_test_label cnn_pred/${event}_pred
done
