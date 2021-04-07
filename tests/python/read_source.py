import os
import sys
import cv2

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/lib.linux-x86_64-3.6'))
sys.path.append(lib_path)
from nvcodec import VideoSource, VideoDecoder, VideoEncoder

source = VideoSource("rtmp://58.200.131.2:1935/livetv/hunantv", 0)
decoder = VideoDecoder()
while True:
    h264_data = source.read()
    frames = decoder.decode(h264_data)
    if frames:
        cv2.imshow("Demo", frames[0])
        cv2.waitKey(1)
