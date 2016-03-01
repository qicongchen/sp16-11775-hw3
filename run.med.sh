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
echo "#       MED with MFCC Features      #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=200
for event in P001 P002 P003; do
  for part in 0 1 2; do
    echo "=========  Event $event Part $part ========="
    # now train a svm model
    python scripts/train_svm.py $event $part "kmeans/" "feat" "dense" $feat_dim_mfcc mfcc_pred/svm.$event.part$part.model || exit 1;
    # apply the svm model to the validation videos;
    # output the score of each validation video to a file ${event}_part${part}_pred 
    python scripts/test_svm.py $event $part mfcc_pred/svm.$event.part$part.model "kmeans/" "feat" "dense" $feat_dim_mfcc mfcc_pred/${event}_part${part}_pred || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_part${part}_test_label mfcc_pred/${event}_part${part}_pred
  done
done

echo "#####################################"
echo "#       MED with ASR Features      #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=12760
for event in P001 P002 P003; do
  for part in 0 1 2; do
    echo "=========  Event $event Part $part ========="
    # now train a svm model
    python scripts/train_svm.py $event $part "asr_feat/asr_bof/" "bof" "sparse" $feat_dim_asr asr_pred/svm.$event.part$part.model || exit 1;
    # apply the svm model to the validation videos;
    # output the score of each validation video to a file ${event}_part${part}_pred 
    python scripts/test_svm.py $event $part asr_pred/svm.$event.part$part.model "asr_feat/asr_bof/" "bof" "sparse" $feat_dim_asr asr_pred/${event}_part${part}_pred || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_part${part}_test_label asr_pred/${event}_part${part}_pred
  done
done

echo "#####################################"
echo "#       MED with CNN Features      #"
echo "#####################################"
mkdir -p cnn_pred
# iterate over the events
feat_dim_cnn=4096
for event in P001 P002 P003; do
  for part in 0 1 2; do
    echo "=========  Event $event Part $part ========="
    # now train a svm model
    python scripts/train_svm.py $event $part "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/svm.$event.part$part.model || exit 1;
    # apply the svm model to the validation videos;
    # output the score of each validation video to a file ${event}_part${part}_pred 
    python scripts/test_svm.py $event $part cnn_pred/svm.$event.part$part.model "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/${event}_part${part}_pred || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_part${part}_test_label cnn_pred/${event}_part${part}_pred
  done
done

echo "#####################################"
echo "#       MED with fusion Features      #"
echo "#####################################"
mkdir -p fusion_pred
# iterate over the events
for event in P001 P002 P003; do
  for part in 0 1 2; do
    echo "=========  Event $event Part $part ========="
    # apply the late fusion;
    # output the score of each validation video to a file ${event}_part${part}_pred 
    python scripts/late_fusion.py $event $part fusion_pred/${event}_part${part}_pred || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_part${part}_test_label fusion_pred/${event}_part${part}_pred
  done
done

echo "#####################################"
echo "#       MED final prediction      #"
echo "#####################################"
# iterate over the events
for event in P001 P002 P003; do
  echo "=========  Event $event ========="
  # now train a svm model
  python scripts/train_svm.py $event "-1" "kmeans/" "feat" "dense" $feat_dim_mfcc mfcc_pred/svm.$event.model || exit 1;
  # apply the svm model to the test videos;
  # output the score of each test video to a file ${event}_pred 
  python scripts/test_svm.py $event "-1" mfcc_pred/svm.$event.model "kmeans/" "feat" "dense" $feat_dim_mfcc mfcc_pred/${event}_pred || exit 1;
done
# iterate over the events
for event in P001 P002 P003; do
  echo "=========  Event $event ========="
  # now train a svm model
  python scripts/train_svm.py $event "-1" "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/svm.$event.model || exit 1;
  # apply the svm model to the test videos;
  # output the score of each test video to a file ${event}_pred 
  python scripts/test_svm.py $event "-1" cnn_pred/svm.$event.model "cnn/" "feat" "dense" $feat_dim_cnn cnn_pred/${event}_pred || exit 1;
done
# iterate over the events
for event in P001 P002 P003; do
  echo "=========  Event $event ========="
  # apply the late fusion;
  # output the score of each validation video to a file ${event}_pred 
  python scripts/late_fusion.py $event -1 fusion_pred/${event}_pred || exit 1;
done
