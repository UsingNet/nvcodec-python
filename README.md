NvCodec - Python
---------------------------

## Require
* cuda >= 11.2
* numpy >= 1.7
* python >= 3.6
* gcc >= 7.5
* make >= 4.1

## Install
```shell
pip install pynvcodec
```

## Usage

### 0. Init PyNvCodec
```python
from nvcodec import VideoSource, VideoDecoder, VideoEncoder
```

### 1. Use VideoSource

```python
source = VideoSource("rtmp://RTMP_URL")
h264_data = source.read()
```

#### 2. Use VideoDecoder
```python
decoder = VideoDecoder()
frames = decoder.decode(h264_data)
# frames can be NULL or List<frame>
```

#### 3. Use VideoEncoder
```python
encoder = VideoEncoder(width, height)
h264_data = encoder.encode(frame)
```
