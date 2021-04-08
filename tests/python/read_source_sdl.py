import os
import sys
from cv2 import cv2
import sdl2
import sdl2.ext
import numpy
import time

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/lib.linux-x86_64-3.6'))
sys.path.append(lib_path)
from nvcodec import VideoSource, VideoDecoder, VideoEncoder

windowArray = None
window = None

def showImage(image):
    global windowArray, window
    if windowArray is None:
        sdl2.ext.init()
        window = sdl2.ext.Window("test", size=(image.shape[0],image.shape[1]))
        window.show()
        windowSurf = sdl2.SDL_GetWindowSurface(window.window)
        windowArray = sdl2.ext.pixels3d(windowSurf.contents)
    numpy.copyto(windowArray, image)
    window.refresh()


# source = VideoSource("rtmp://58.200.131.2:1935/livetv/hunantv")
source = VideoSource("/tmp/1.mp4")
decoder = VideoDecoder()
while True:
    h264_data = source.read()
    if not h264_data:
        break
    frames = decoder.decode(h264_data, 1)
    for frame in frames:
        showImage(frame)