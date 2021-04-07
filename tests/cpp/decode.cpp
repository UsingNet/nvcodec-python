#include "source.h"
#include "decoder.h"
#include "stdio.h"

#ifndef TEST_RTMP_URL
#define TEST_RTMP_URL "rtmp://58.200.131.2:1935/livetv/hunantv"
#endif

int main(int argc, char** argv){
    videoSourceHandle videoSource = videoSource_init( (char*)TEST_RTMP_URL, 0);
    // videoSourceHandle videoSource = videoSource_init((char*)"rtmp://10.10.1.108:8981/app/video/001", 1);
    AVPacket packet;
    int width, height, size;
    cudaVideoSurfaceFormat format;
    videoDecoderHandle videoDecode = videoDecoder_init(AV_CODEC_ID_H264);
    videoFrameList* frameList;
    while(1){
        if(videoSource_read(videoSource, &packet)<0){
            break;
        }
        frameList = videoDecoder_decode(videoDecode, packet.data, packet.size);
        if(frameList!=NULL){
            printf("Decode Frame %dx%d, Frames %d\n", frameList->width, frameList->height, frameList->length);
        }
        videoFrameList_destory(&frameList);
    }
    videoDecoder_destroy(videoDecode);
    return 0;
}