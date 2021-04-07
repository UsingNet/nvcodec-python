#include "source.h"
#include "stdio.h"

#ifndef TEST_RTMP_URL
#define TEST_RTMP_URL "rtmp://58.200.131.2:1935/livetv/hunantv"
#endif

int main(int argc, char** argv){
    videoSourceHandle videoSource = videoSource_init( (char*)TEST_RTMP_URL, 0);
    // videoSourceHandle videoSource = videoSource_init((char*)"rtmp://10.10.1.108:8981/app/video/001", 1);
    AVPacket packet;
#ifdef DEBUG
    FILE* fp = fopen("/tmp/save.h264", "wb");
#endif
    while(1){
        if(videoSource_read(videoSource, &packet)<0){
            break;
        }
#ifdef DEBUG
        fwrite(packet.data, 1, packet.size, fp);
#endif
        printf("Read AVPacket Index: %d Size:%d\n", packet.stream_index, packet.size);
    }
#ifdef DEBUG
    fclose(fp);
#endif
    return 0;
}