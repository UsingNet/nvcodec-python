#ifndef NVCODEC_PYTHON_VIDEO_ENCODER_H
#define NVCODEC_PYTHON_VIDEO_ENCODER_H

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
    void* enc;
}videoEncoder;

typedef videoEncoder* videoEncoderHandle;


typedef struct
{
    uint8_t* data;
    int size;
}videoEncodedBuffer;


videoEncoderHandle videoEncoder_init(int width, int height);
int videoEncoder_destroy(videoEncoderHandle handle);
videoEncodedBuffer* videoEncoder_encode(videoEncoderHandle handle, u_int8_t* in);
videoEncodedBuffer* videoEncoder_encode_end(videoEncoderHandle handle);
void videoEncodedBuffer_destory(videoEncodedBuffer**);
videoEncodedBuffer* videoEncodedBuffer_init(int size);


#endif