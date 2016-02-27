import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} file_list output".format(sys.argv[0])
        print "file_list -- the list of videos"
        print "output -- the image list"
        exit(1)

    file_list = sys.argv[1]
    output = sys.argv[2]

    fread = open(file_list, "r")
    fwrite = open(output, "w")

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        frame_dir = "frame/" + video_id + "/"
        if os.path.exists(frame_dir) is False:
            continue
        for frame_file in os.listdir(frame_dir):
            if '.jpeg' not in frame_file:
                continue
            frame_id = frame_file.split('.jpeg')[0]
            image_path = frame_dir+frame_file
            fwrite.write("%s\n" % image_path)
    fread.close()
    fwrite.close()

    print "Image list extracted successfully!"
