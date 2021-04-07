CFLAGS=-Wall -O2 -lstdc++ -pthread -lm -fPIC
ifdef DEBUG
CFLAGS=-g -Wall -O0 -D DEBUG -lstdc++ -pthread -lm -fPIC
endif

ifndef CUDA_PATH
CUDA_PATH=/usr/local/cuda
endif

ifndef PYTHON_VERSION
PYTHON_VERSION=$(shell python3 -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor,))")
endif

ifndef PYTHON_INCLUDE_PATH
PYTHON_INCLUDE_PATH=/usr/include/python${PYTHON_VERSION}
endif

ifndef PYTHON_BIN
PYTHON_BIN=python${PYTHON_VERSION}
endif

ifndef PREFIX
PREFIX=/usr/local
ifdef VIRTUAL_ENV
PREFIX=${VIRTUAL_ENV}
endif
endif

lib: build/lib/libnvcodec.a
test: build/tests/read_source build/tests/decode build/tests/encode

python: lib
	${PYTHON_BIN} setup.py build

out:
	mkdir -p build/tests
	mkdir -p build/lib

SRC_FILES=$(wildcard src/*.cpp) $(wildcard src/cuvid/*.cpp) $(wildcard src/cuvid/NvDecoder/*.cpp) $(wildcard src/cuvid/Utils/*.cpp) src/cuvid/NvEncoder/NvEncoder.cpp src/cuvid/NvEncoder/NvEncoderCuda.cpp
OPENCV_LIB=-I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui

lib_cuda: build/lib/libcolor_space.a

build/lib/libcolor_space.a: src/cuvid/Utils/ColorSpace.cu out
	nvcc -DCUDNN  --compiler-options "-fPIC -lstdc++ -pthread -lm" -c src/cuvid/Utils/ColorSpace.cu -o build/lib/libcolor_space.a


FLAGS=-L${CUDA_PATH}/lib64 -Lbuild/lib -lavformat -lavcodec -lavutil -lcudart -lnvcuvid -lnvidia-encode -lcuda -Isrc -I${CUDA_PATH}/include -Isrc/cuvid ${CFLAGS} 

# build/lib/libnvcodec.so: lib_cuda out
# 	g++ -o build/lib/libnvcodec.so -shared ${SRC_FILES} -lcolor_space ${FLAGS} -fPIC


build/tests/read_source: tests/cpp/read_source.cpp lib
	g++ -o build/tests/read_source tests/cpp/read_source.cpp -lnvcodec ${FLAGS}

build/lib/libnvcodec.a: lib_cuda out
    # g++ -o build/lib/libnvcodec.so -shared ${SRC_FILES} -lcolor_space ${FLAGS} -fPIC
	mkdir -p build/object
	cd build/object; g++ -c ../../src/*.cpp ../../src/**/*.cpp ../../src/**/**/*.cpp -I../../src -I../../src/cuvid -I${CUDA_PATH}/include
	ar rcs build/lib/libnvcodec.a build/object/*.o


build/tests/decode: tests/cpp/decode.cpp lib
	g++ -o build/tests/decode tests/cpp/decode.cpp -lnvcodec  ${OPENCV_LIB} ${FLAGS}

build/tests/encode: tests/cpp/encode.cpp lib
	g++ -o build/tests/encode tests/cpp/encode.cpp -lnvcodec ${OPENCV_LIB} ${FLAGS}

clean:
	rm build -rf
	rm pynvcodec.egg-info -rf
	rm dist -rf

python-interface:
	${PYTHON_BIN} setup.py build

release: clean python-interface
	${PYTHON_BIN} setup.py sdist
	${PYTHON_BIN} -m twine upload dist/*
