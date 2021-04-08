#ifndef NVCODEC_PYTHON_VIDEO_DECODER_H
#define NVCODEC_PYTHON_VIDEO_DECODER_H



#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include "cuvid/nvcuvid.h"
#include <libavcodec/avcodec.h>


#ifdef __cplusplus
}
#endif

typedef struct
{
    CUcontext cuContext;
    void* dec;
}videoDecoder;

typedef videoDecoder* videoDecoderHandle;

typedef struct
{
    uint8_t* pFrames;
    int perFrameSize;
    int width;
    int height;
    int length;
}videoFrameList;



videoDecoderHandle videoDecoder_init(enum AVCodecID codec);
int videoDecoder_destroy(videoDecoderHandle handle);
videoFrameList* videoDecoder_decode(videoDecoderHandle handle, u_int8_t* in, size_t in_size, char*error);
void videoFrameList_destory(videoFrameList**);
videoFrameList* videoFrameList_init(int width, int height, int length);


#endif