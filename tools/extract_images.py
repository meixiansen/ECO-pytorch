
import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process

import argparse
out_path ='../dataset/ucf101'


def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(fcount):
        ret, frame = video.read()
        try:
            assert ret
            cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
            access_path = '{}/{:06d}.jpg'.format(vid_name, i)
            file_list.append(access_path)
        except:
            i=i-1
    print ('{} done'.format(vid_name))
    sys.stdout.flush()
    return file_list



if __name__ == '__main__':
  src_path = '../data/ucf'
  ext = 'avi'
  vid_list = glob.glob(src_path+'/*/*.'+ext)
  #print (vid_list)
  for vidname in vid_list:    
      dump_frames(vidname)
  
