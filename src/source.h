#ifndef NVCODEC_PYTHON_VIDEO_SOURCE_H
#define NVCODEC_PYTHON_VIDEO_SOURCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif


typedef struct
{
    AVFormatContext *pFormatCtx;
    AVDictionary *options;
    char* url;
    AVBSFContext *bsfc;
    int video_stream_index;
}videoSource;

typedef videoSource* videoSourceHandle;

enum AVPixelFormat videoSource_getAVPixelFormat(videoSourceHandle handle);
enum AVCodecID videoSource_getVideoCodecId(videoSourceHandle handle);
videoSourceHandle videoSource_init(char* url, int listen);
int videoSource_destroy(videoSourceHandle handle);
int videoSource_connect(videoSourceHandle handle);
int videoSource_read(videoSourceHandle handle, AVPacket* packet);
int videoSource_isConnect(videoSourceHandle handle);
int videoSource_close(videoSourceHandle handle);


#endif //NVCODEC_PYTHON_VIDEO_SOURCE_H
