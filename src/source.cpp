#include <stdio.h>
#include "source.h"

videoSourceHandle videoSource_init(char* url, int listen){
#ifdef DEBUG
    av_log_set_level(AV_LOG_TRACE);
#else
    av_log_set_level(AV_LOG_FATAL);
#endif
    av_register_all();
    avformat_network_init();
    videoSourceHandle handle = (videoSourceHandle)malloc(sizeof(videoSource));
    handle->url = (char*)malloc(strlen(url));
    memcpy(handle->url, url, strlen(url));
    handle->bsfc = NULL;
    handle->options = NULL;
    handle->pFormatCtx = NULL;
    handle->video_stream_index = 0;
    
    av_bsf_alloc(av_bsf_get_by_name("h264_mp4toannexb"), &(handle->bsfc));

    if(listen){
        av_dict_set(&(handle->options), "listen", "1", 0);
    }
    // av_dict_set(&(handle->options), "timeout", "3", 0);

    return handle;
}

int videoSource_destroy(videoSourceHandle handle){
    if(handle->pFormatCtx != NULL){
        videoSource_close(handle);
    }
    if(handle->url != NULL){
        free(handle->url);
        handle->url = NULL;
    }
    if(handle->bsfc != NULL){
        av_bsf_free(&(handle->bsfc));
        handle->bsfc = NULL;
    }
    free(handle);
    return 0;
}

int videoSource_connect(videoSourceHandle handle){
    if(handle->pFormatCtx != NULL){
        return 1;
    }

    char error_buf[128] = {0};
    int error_code;
    if ((error_code = avformat_open_input(&(handle->pFormatCtx), handle->url, NULL, &(handle->options))) < 0){
        // AVERROR_INPUT_CHANGED
        av_make_error_string(error_buf, 127, error_code);
        printf("RTMP/Listen %d: %s\n", error_code, error_buf);
        // throw RTMPException(error_buf, 301);
        return -301;
    }
    if ((error_code = avformat_find_stream_info(handle->pFormatCtx, NULL))<0)
    {
        printf("Couldn't find stream information\n");
        return -302;
    }
    unsigned int i;
    for (i = 0; i< handle->pFormatCtx->nb_streams; i++){
        if (handle->pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            handle->video_stream_index = i;
            avcodec_parameters_copy(handle->bsfc->par_in,handle->pFormatCtx->streams[i]->codecpar);
            av_bsf_init(handle->bsfc);
            break;
        }
    }
    if (handle->pFormatCtx->nb_streams == i)
    {
        printf("Didn't find a video stream\n");
        // throw RTMPException("Didn't find a video stream", 303);
        return -303;
    }
    return 0;
}


int videoSource_read(videoSourceHandle handle, AVPacket* packet){
    if(handle->pFormatCtx == NULL){
        if(videoSource_connect(handle)<0){
            return -1;
        }
    }
    while(av_read_frame(handle->pFormatCtx, packet)==0){
        av_bsf_send_packet(handle->bsfc, packet);
        av_bsf_receive_packet(handle->bsfc, packet);
        if(packet->size>0 && packet->stream_index == handle->video_stream_index){
            return packet->size;
        }
    }
    return -1;
}

int videoSource_close(videoSourceHandle handle){
    if( handle->pFormatCtx != NULL) {
        avformat_close_input(&(handle->pFormatCtx));
        av_free(handle->pFormatCtx);
        handle->pFormatCtx = NULL;
    }
    return 0;
}

enum AVPixelFormat videoSource_getAVPixelFormat(videoSourceHandle handle){
    if(handle->pFormatCtx != NULL) {
        return (AVPixelFormat)(handle->pFormatCtx->streams[handle->video_stream_index]->codecpar->format);
    }else{
        return AV_PIX_FMT_NONE;
    }
}

enum AVCodecID videoSource_getVideoCodecId(videoSourceHandle handle){
    if(handle->pFormatCtx != NULL) {
        return handle->pFormatCtx->video_codec_id;
    }else{
        return AV_CODEC_ID_NONE;
    }
}

int videoSource_isConnect(videoSourceHandle handle){
    return handle->pFormatCtx != NULL;
}
