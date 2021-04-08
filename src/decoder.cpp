#include "decoder.h"
#include "cuvid/Utils/NvCodecUtils.h"
#include "cuvid/Utils/FFmpegDemuxer.h"
#include "cuvid/Utils/ColorSpace.h"
#include "cuvid/AppDecUtils.h"
#include <libavcodec/avcodec.h>
#include "cuvid/Utils/Logger.h"
#include <string.h>
#include "cuvid/NvDecoder/NvDecoder.h"
#include <cuda_runtime.h>

#define DEC(handle) ((NvDecoder*)(handle->dec))


videoDecoderHandle videoDecoder_init(enum AVCodecID codec){
    videoDecoderHandle handle = (videoDecoderHandle)malloc(sizeof(videoDecoder));
    ck(cuInit(0));
    handle->cuContext = nullptr;
    createCudaContext(&(handle->cuContext), 0, 0);
    handle->dec = (void*)(new NvDecoder(handle->cuContext, false, FFmpeg2NvCodecId(codec)));
    return handle;
}

int videoDecoder_destroy(videoDecoderHandle handle){
    delete(handle->dec);
    cuCtxDestroy(handle->cuContext);
    handle->cuContext = nullptr;
    handle->dec = nullptr;
    return 0;
}

void videoFrameList_destory(videoFrameList** list){
    if(*list != NULL){
        if((*list)->pFrames != NULL){
            free((*list)->pFrames);
            (*list)->pFrames = NULL;
        }
        free((*list));
        *list = NULL;
    }
}

videoFrameList* videoFrameList_init(int width, int height, int length){
    videoFrameList* frameList = (videoFrameList*)malloc(sizeof(videoFrameList));
    frameList->height = height;
    frameList->width = width;
    frameList->length = length;
    frameList->perFrameSize = height*width*4;
    frameList->pFrames = (uint8_t*)malloc(frameList->height * frameList->width * 4 * frameList->length);
    return frameList;
}


videoFrameList* videoDecoder_decode(videoDecoderHandle handle, u_int8_t* in, size_t in_size, char*error){
    int nFrameReturned;
    int i;
    int frameSize;
    uint8_t *pVideo = NULL, *pFrame;
    videoFrameList* frameList = NULL;
    CUdeviceptr dpFrame = 0, nv12Frame = 0;
    if(error!=NULL){
        error[0] = NULL;
    }
    try{
        nFrameReturned = DEC(handle)->Decode(in, in_size);
    }catch(NVDECException e){
        if(error != NULL){
            strcpy(error, e.what());
        }
        return NULL;
    }
    for (i = 0; i < nFrameReturned; i++) {
        pFrame = DEC(handle)->GetFrame();
        frameSize = DEC(handle)->GetFrameSize();
        if(i == 0){
            frameList = videoFrameList_init(DEC(handle)->GetWidth(), DEC(handle)->GetHeight(), nFrameReturned);
            ck(cuMemAlloc(&dpFrame, frameList->width * frameList->height * 4));
            ck(cuMemAlloc(&nv12Frame, frameSize));
        }
        cudaMemcpy((void*)nv12Frame, pFrame, frameSize, cudaMemcpyHostToDevice);
        Nv12ToColor32<BGRA32>((uint8_t*)nv12Frame, frameList->width, (uint8_t *)dpFrame, 4 * frameList->width, frameList->width, frameList->height);
        int output_size = frameList->width * frameList->height * 4;
        cudaMemcpy((void*)(frameList->pFrames+i*frameList->perFrameSize), (uint8_t*)dpFrame, output_size, cudaMemcpyDeviceToHost);
    }
    cuMemFree(dpFrame);
    return frameList;
}