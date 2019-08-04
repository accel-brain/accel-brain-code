# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from setuptools import Extension
import numpy as np
import os

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cython_flag = True
except ImportError:
    cython_flag = False


if cython_flag is True:
    file_format = ".pyx"
    cmdclass = {"build_ext": build_ext}
else:
    file_format = ".c"
    cmdclass = {}


pyx_list = []
for dirpath, dirs, files in os.walk('.'):
    for f in files:
        if file_format in f and "checkpoint" not in f:
            pyx_path = os.path.join(dirpath, f)
            pyx_list.append(Extension("*", [pyx_path]))


if cython_flag is True:
    ext_modules = cythonize(pyx_list, include_path=[np.get_include()])
else:
    ext_modules = pyx_list


def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(
    name='pydbm',
    version='1.5.0',
    description='`pydbm` is Python library for building Restricted Boltzmann Machine(RBM), Deep Boltzmann Machine(DBM), Long Short-Term Memory Recurrent Temporal Restricted Boltzmann Machine(LSTM-RTRBM), and Shape Boltzmann Machine(Shape-BM). From the view points of functionally equivalents and structural expansions, this library also prototypes many variants such as Encoder/Decoder based on LSTM, Convolutional Auto-Encoder, and Spatio-temporal Auto-Encoder.',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern',
    author='chimera0',
    author_email='ai-brain-lab@accel-brain.com',
    license='GPL2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='restricted boltzmann machine autoencoder auto-encoder rnn rbm rtrbm convolution deconvolution spatio-temporal encoder decoder LSTM',
    install_requires=['numpy', 'cython'],
    include_dirs=[ '.', np.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(pyx_list, include_path=[np.get_include()])
)
