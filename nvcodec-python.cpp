#include <stdio.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "source.h"
#include "decoder.h"
#include "encoder.h"


typedef struct
{
    PyObject_HEAD
    long long m_handle;
}NvCodec;

static PyMemberDef NvCodec_DataMembers[] =
{
        {(char*)"m_handle",   T_LONGLONG, offsetof(NvCodec, m_handle),   0, (char*)"NvCodec handle ptr"},
        {NULL, 0, 0, 0, NULL}
};

/* ----------- VideoSource Part --------------- */

static PyObject* VideoSource_read(NvCodec* Self)
{
    videoSourceHandle m_handle = (videoSourceHandle)Self->m_handle;
    AVPacket *packet = av_packet_alloc();
    if(videoSource_read(m_handle, packet) < 0){
        av_packet_free(&packet);
        return Py_None;
    }
    PyObject* rtn = PyBytes_FromStringAndSize((const char*)packet->data, packet->size);
    av_packet_free(&packet);
    return rtn;
}

static PyMethodDef VideoSource_MethodMembers[] =
{
        {"read", (PyCFunction)VideoSource_read, METH_NOARGS,  "read h264 from video source"},
        {NULL, NULL, 0, NULL}
};

static void VideoSource_Destruct(NvCodec* Self)
{
    videoSourceHandle m_handle = (videoSourceHandle)(Self->m_handle);
    videoSource_destroy(m_handle);
    Py_TYPE(Self)->tp_free((PyObject*)Self);
}


static PyObject* VideoSource_Str(NvCodec* Self)
{
    return Py_BuildValue("s", "<nvcodec-python.VideoSource>");
}

static PyObject* VideoSource_Repr(NvCodec* Self)
{
    return VideoSource_Str(Self);
}

static void VideoSource_init(NvCodec* Self, PyObject* pArgs)
{
    unsigned char* url;
    unsigned int listen = 0;
    if(!PyArg_ParseTuple(pArgs, "s|I", &url, &listen)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass url string!");
        return;
    }

    Self->m_handle = (long long)(videoSource_init((char*)url, listen));
}

static PyTypeObject VideoSource_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)"NvCodec.VideoSource",
        sizeof(NvCodec),
        0,
        (destructor)VideoSource_Destruct,
        NULL,NULL,NULL,NULL,
        (reprfunc)VideoSource_Repr,
        NULL,NULL,NULL,NULL,NULL,
        (reprfunc)VideoSource_Str,
        NULL,NULL,NULL,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "NvCodec Python VideoSource Objects --- Extensioned by nvcodec",
        NULL,NULL,NULL,0,NULL,NULL,
        VideoSource_MethodMembers,
        NvCodec_DataMembers,
        NULL,NULL,NULL,NULL,NULL,0,
        (initproc)VideoSource_init,
        NULL,
};

/* ----------- Decoder Part --------------- */

static PyObject* VideoDecoder_decode(NvCodec* Self, PyObject* pArgs)
{
    videoDecoderHandle m_handle = (videoDecoderHandle)Self->m_handle;
    
    unsigned char* data;
    int len;
    unsigned int type = 0;
    if(!PyArg_ParseTuple(pArgs, "y#|I", &data, &len, &type)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should video byte data!");
        return Py_None;
    }
    
    PyObject* rtn = Py_BuildValue("[]");
    char error_str[128];
    videoFrameList* list = videoDecoder_decode(m_handle, data, len, error_str);
    if(list == NULL){
        if(error_str[0] != NULL){
            PyErr_Format(PyExc_ValueError, "%s", error_str);
            return NULL;
        }
        return rtn;
    }

    npy_intp dims[3] = {(npy_intp)(list->height), (npy_intp)(list->width), 4};
    PyObject* tempFrame;
    for(int i = 0;i<list->length;i++){
        tempFrame = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, list->pFrames + (i*(list->perFrameSize)));
        PyArray_ENABLEFLAGS((PyArrayObject*) tempFrame, NPY_ARRAY_OWNDATA);
        if(type != 0){
            tempFrame = PyArray_SwapAxes((PyArrayObject*)tempFrame, 0, 1);
        }
        PyList_Append(rtn, tempFrame);
    }
    videoFrameList_destory(&list);
    return rtn;
}

static PyMethodDef VideoDecoder_MethodMembers[] =
{
        {"decode", (PyCFunction)VideoDecoder_decode, METH_VARARGS,  "decode video frame"},
        {NULL, NULL, 0, NULL}
};

static void VideoDecoder_Destruct(NvCodec* Self)
{
    videoDecoderHandle m_handle = (videoDecoderHandle)(Self->m_handle);
    videoDecoder_destroy(m_handle);
    Py_TYPE(Self)->tp_free((PyObject*)Self);
}


static PyObject* VideoDecoder_Str(NvCodec* Self)
{
    return Py_BuildValue("s", "<nvcodec-python.VideoDecoder>");
}

static PyObject* VideoDecoder_Repr(NvCodec* Self)
{
    return VideoDecoder_Str(Self);
}

static void VideoDecoder_init(NvCodec* Self, PyObject* pArgs)
{
    unsigned int format = AV_CODEC_ID_H264;
    if(!PyArg_ParseTuple(pArgs, "|I", &format)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass AV_CODEC_ID!");
        return;
    }
    Self->m_handle = (long long)(videoDecoder_init((enum AVCodecID)format));
}

static PyTypeObject VideoDecoder_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)"NvCodec.VideoDecoder",
        sizeof(NvCodec),
        0,
        (destructor)VideoDecoder_Destruct,
        NULL,NULL,NULL,NULL,
        (reprfunc)VideoDecoder_Repr,
        NULL,NULL,NULL,NULL,NULL,
        (reprfunc)VideoDecoder_Str,
        NULL,NULL,NULL,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "NvCodec Python VideoDecoder Objects --- Extensioned by nvcodec",
        NULL,NULL,NULL,0,NULL,NULL,
        VideoDecoder_MethodMembers,
        NvCodec_DataMembers,
        NULL,NULL,NULL,NULL,NULL,0,
        (initproc)VideoDecoder_init,
        NULL,
};

/* ----------- Encoder Part --------------- */

static PyObject* VideoEncoder_encode(NvCodec* Self, PyObject* pArgs)
{
    PyArrayObject *vecin;
    if (!PyArg_ParseTuple(pArgs, "O!", &PyArray_Type, &vecin)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass ABGR image numpy array!");
        return NULL;
    }

    if (NULL == vecin){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(vecin) != 4){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass ABGR image numpy array by height*width*channel!");
        return NULL;
    }

    videoEncoderHandle m_handle = (videoEncoderHandle)Self->m_handle;

    PyObject* bytes = PyObject_CallMethod((PyObject*)vecin, "tobytes", NULL);
    
    int length;
    unsigned char* data;
    PyArg_Parse(bytes, "y#", &data, &length);

    videoEncodedBuffer* buffer = videoEncoder_encode(m_handle, data);
    if(buffer == NULL){
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* rtn = PyBytes_FromStringAndSize((const char*)buffer->data, buffer->size);
    videoEncodedBuffer_destory(&buffer);
    return rtn;
}

static PyMethodDef VideoEncoder_MethodMembers[] =
{
        {"encode", (PyCFunction)VideoEncoder_encode, METH_VARARGS,  "encode video frame"},
        {NULL, NULL, 0, NULL}
};

static void VideoEncoder_Destruct(NvCodec* Self)
{
    videoEncoderHandle m_handle = (videoEncoderHandle)(Self->m_handle);
    videoEncoder_destroy(m_handle);
    Py_TYPE(Self)->tp_free((PyObject*)Self);
}


static PyObject* VideoEncoder_Str(NvCodec* Self)
{
    return Py_BuildValue("s", "<nvcodec-python.VideoEncoder>");
}

static PyObject* VideoEncoder_Repr(NvCodec* Self)
{
    return VideoEncoder_Str(Self);
}

static void VideoEncoder_init(NvCodec* Self, PyObject* pArgs)
{
    unsigned int width,height;
    if(!PyArg_ParseTuple(pArgs, "II", &width, &height)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass width and height!");
        return;
    }
    Self->m_handle = (long long)(videoEncoder_init(width, height));
}

static PyTypeObject VideoEncoder_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)"NvCodec.VideoEncoder",
        sizeof(NvCodec),
        0,
        (destructor)VideoEncoder_Destruct,
        NULL,NULL,NULL,NULL,
        (reprfunc)VideoEncoder_Repr,
        NULL,NULL,NULL,NULL,NULL,
        (reprfunc)VideoEncoder_Str,
        NULL,NULL,NULL,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "NvCodec Python VideoEncoder Objects --- Extensioned by nvcodec",
        NULL,NULL,NULL,0,NULL,NULL,
        VideoEncoder_MethodMembers,
        NvCodec_DataMembers,
        NULL,NULL,NULL,NULL,NULL,0,
        (initproc)VideoEncoder_init,
        NULL,
};




void NvCodec_module_destroy(void *_){
    // Pass
}

static PyModuleDef ModuleInfo =
{
        PyModuleDef_HEAD_INIT,
        "NvCodec Module",
        "NvCodec by NvCodec",
        -1,
        NULL, NULL, NULL, NULL,
        NvCodec_module_destroy
};

PyMODINIT_FUNC
PyInit_nvcodec(void) {    
    PyObject * pReturn = NULL;

    VideoSource_ClassInfo.tp_new = PyType_GenericNew;
    if(PyType_Ready(&VideoSource_ClassInfo) < 0)
        return NULL;
    
    VideoDecoder_ClassInfo.tp_new = PyType_GenericNew;
    if(PyType_Ready(&VideoDecoder_ClassInfo) < 0)
        return NULL;
    
    VideoEncoder_ClassInfo.tp_new = PyType_GenericNew;
    if(PyType_Ready(&VideoEncoder_ClassInfo) < 0)
        return NULL;

    pReturn = PyModule_Create(&ModuleInfo);
    if(pReturn == NULL)
        return NULL;

    Py_INCREF(&ModuleInfo);
    PyModule_AddObject(pReturn, "VideoSource", (PyObject*)&VideoSource_ClassInfo);
    PyModule_AddObject(pReturn, "VideoDecoder", (PyObject*)&VideoDecoder_ClassInfo);
    PyModule_AddObject(pReturn, "VideoEncoder", (PyObject*)&VideoEncoder_ClassInfo);
    import_array();
    return pReturn;
}
