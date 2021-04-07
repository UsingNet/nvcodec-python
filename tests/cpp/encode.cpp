#include "source.h"
#include "decoder.h"
#include "encoder.h"
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
    videoEncoderHandle videoEncode = NULL;
    videoFrameList* frameList;
    videoEncodedBuffer* buffer;
#ifdef DEBUG
    FILE* fp = fopen("/tmp/encode.h264", "wb");
#endif
    while(1){
        if(videoSource_read(videoSource, &packet)<0){
            break;
        }
        frameList = videoDecoder_decode(videoDecode, packet.data, packet.size);
        if(frameList==NULL){
            continue;
        }
        if(videoEncode==NULL){
            videoEncode = videoEncoder_init(frameList->width, frameList->height);
        }
        buffer = videoEncoder_encode(videoEncode, frameList->pFrames);
        videoFrameList_destory(&frameList);
        if(buffer == NULL){
            continue;
        }
        printf("Encode Buffer size: %d\n", buffer->size);
#ifdef DEBUG
        fwrite(buffer->data, 1, buffer->size, fp);
#endif
        videoEncodedBuffer_destory(&buffer);
    }
#ifdef DEBUG
    fclose(fp);
#endif
    videoDecoder_destroy(videoDecode);
    videoEncoder_destroy(videoEncode);
    return 0;
}