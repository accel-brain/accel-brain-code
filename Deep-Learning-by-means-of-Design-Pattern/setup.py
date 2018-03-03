# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from setuptools import Extension
import numpy as np
import os
from Cython.Distutils import build_ext
from Cython.Build import cythonize


pyx_list = []
for dirpath, dirs, files in os.walk('.'):
    for f in files:
        if ".pyx" in f:
            pyx_path = os.path.join(dirpath, f)
            pyx_list.append(Extension("*", [pyx_path]))

setup(
    name='pydbm',
    version='1.1.8',
    description='pydbm is Python library for building restricted boltzmann machine, deep boltzmann machine, and multi-layer neural networks.',
    long_description='The models are functionally equivalent to stacked auto-encoder. The main function I observe is the same as dimensions reduction(or pre-training).',
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
    keywords='restricted boltzmann machine autoencoder auto-encoder rnn rbm rtrbm',
    install_requires=['numpy', 'cython'],
    include_dirs=[ '.', np.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(pyx_list, include_path=[np.get_include()])
)
