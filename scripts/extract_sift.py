import cv2
import numpy
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: {0} file_list".format(sys.argv[0])
        print "file_list -- the list of videos"
        exit(1)

    file_list = sys.argv[1]

    fread = open(file_list, "r")

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        frame_dir = "frame/" + video_id + "/"
        sift_dir = "sift/" + video_id + "/"
        if os.path.exists(frame_dir) is False:
            continue
        for frame_file in os.listdir(frame_dir):
            if '.jpeg' not in frame_file:
                continue
            frame_id = frame_file.split('.jpeg')[0]
            img = cv2.imread(frame_dir+frame_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray, None)

            if des is None:
                continue
            numpy.savetxt(sift_dir+frame_id+".sift", des, delimiter=';')
    fread.close()

    print "Sift features extracted successfully!"
