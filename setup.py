# #!/usr/bin/env python
import sys
import os
import glob
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext
import numpy as np

libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), './build/lib'))

class custom_build_ext(build_ext):
    def build_extensions(self):
        os.system('make lib_cuda')
        build_ext.build_extensions(self)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

nvcodec_dir = '/usr/local/lib'
if 'VIRTUAL_ENV' in os.environ:
    nvcodec_dir = os.path.join(os.environ['VIRTUAL_ENV'], 'lib')

sources = ['nvcodec-python.cpp'] + glob.glob('src/**/*.cpp', recursive=True)

module = Extension('nvcodec', sources=sources, language='c++', 
include_dirs=['src', 'src/cuvid', '/usr/local/cuda/include',np.get_include(),], 
library_dirs=['build/lib', '/usr/local/cuda-11.2/targets/x86_64-linux/lib'],
libraries=['avformat', 'avcodec','avutil','nvcuvid','nvidia-encode','cuda', 'stdc++', 'm', 'cudart', 'color_space'],
)

from distutils.core import setup, Extension
setup(name='pynvcodec',
    version='0.0.6',
    ext_modules=[module],
    cmdclass={'build_ext': custom_build_ext},
    author="Usingnet",
    author_email="developer@usingnet.com",
    license="MIT",
    description="Python interface for nvcodec. Encode/Decode H264 with Nvidia GPU Hardware Acceleration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UsingNet/nvcodec-python",
    # packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    keywords=[
        "pynvcodec",
        "nvcodec",
        "h264",
        "encode",
        "decode",
        "h264 encode",
        "h264 decode",
        "gpu",
        "nvidia"
    ],
    python_requires=">=3.6",
    project_urls={
        'Source': 'https://github.com/UsingNet/nvcodec-python',
        'Tracker': 'https://github.com/UsingNet/nvcodec-python/issues',
    },
    install_requires=['numpy>=1.17']
)