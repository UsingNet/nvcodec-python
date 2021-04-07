#include "encoder.h"
#include "cuvid/NvEncoder/NvEncoderCuda.h"
#include "cuvid/Utils/NvCodecUtils.h"
#include "cuvid/Utils/FFmpegDemuxer.h"
#include "cuvid/Utils/ColorSpace.h"
#include "cuvid/AppDecUtils.h"
#include "cuvid/Utils/NvEncoderCLIOptions.h"
#include "cuvid/nvEncodeAPI.h"

#define ENC(handle) ((NvEncoderCuda*)(handle->enc))

void _InitializeEncoder(NvEncoder* pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

    initializeParams.encodeConfig = &encodeConfig;

    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}


videoEncoderHandle videoEncoder_init(int width, int height){
    videoEncoderHandle handle = (videoEncoderHandle)malloc(sizeof(videoEncoder));
    ck(cuInit(0));
    handle->cuContext = nullptr;
    createCudaContext(&(handle->cuContext), 0, 0);
    handle->enc = new NvEncoderCuda(handle->cuContext, width, height, NV_ENC_BUFFER_FORMAT_ARGB);

    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ARGB;
    NvEncoderInitParam encodeCLIOptions;
    _InitializeEncoder(ENC(handle), encodeCLIOptions, eFormat);
    return handle;
}

int videoEncoder_destroy(videoEncoderHandle handle){
    ENC(handle)->DestroyEncoder();
    delete(ENC(handle));
    cuCtxDestroy(handle->cuContext);
    handle->cuContext = nullptr;
    handle->enc = nullptr;
    return 0;
}

videoEncodedBuffer* videoEncoder_encode_end(videoEncoderHandle handle){
    std::vector<std::vector<uint8_t>> vPacket;
    ENC(handle)->EndEncode(vPacket);
    int currentSize = 0;
    for (std::vector<uint8_t> &packet : vPacket){
        currentSize += packet.size();
    }
    if(currentSize == 0){
        return NULL;
    }
    videoEncodedBuffer* buffer = videoEncodedBuffer_init(currentSize);
    currentSize = 0;
    for (std::vector<uint8_t> &packet : vPacket){
        memcpy(buffer->data+currentSize, reinterpret_cast<char*>(packet.data()), packet.size());
        currentSize+=packet.size();
    }
    return buffer;
}

videoEncodedBuffer* videoEncoder_encode(videoEncoderHandle handle, u_int8_t* in){
    std::vector<std::vector<uint8_t>> vPacket;
    
    const NvEncInputFrame* encoderInputFrame = ENC(handle)->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(handle->cuContext, in, ENC(handle)->GetWidthInBytes(NV_ENC_BUFFER_FORMAT_ARGB,ENC(handle)->GetEncodeWidth()), (CUdeviceptr)encoderInputFrame->inputPtr,
        (int)encoderInputFrame->pitch,
        ENC(handle)->GetEncodeWidth(),
        ENC(handle)->GetEncodeHeight(),
        CU_MEMORYTYPE_HOST, 
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes);
    ENC(handle)->EncodeFrame(vPacket);
    int currentSize = 0;
    for (std::vector<uint8_t> &packet : vPacket){
        currentSize += packet.size();
    }
    if(currentSize == 0){
        return NULL;
    }
    videoEncodedBuffer* buffer = videoEncodedBuffer_init(currentSize);
    currentSize = 0;
    for (std::vector<uint8_t> &packet : vPacket){
        memcpy(buffer->data+currentSize, reinterpret_cast<char*>(packet.data()), packet.size());
        currentSize+=packet.size();
    }
    return buffer;
}

void videoEncodedBuffer_destory(videoEncodedBuffer** buffer){
    if(*buffer == NULL){
        if((*buffer)->data != NULL){
            free((*buffer)->data);
            (*buffer)->data = NULL;
        }
        free(*buffer);
        (*buffer) = NULL;
    }
}

videoEncodedBuffer* videoEncodedBuffer_init(int size){
    videoEncodedBuffer* frame = (videoEncodedBuffer*)malloc(sizeof(videoEncodedBuffer));
    frame->size = size;
    frame->data = (u_int8_t*)malloc(frame->size);
    return frame;
}
