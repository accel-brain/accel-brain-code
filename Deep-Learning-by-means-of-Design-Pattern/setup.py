# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from setuptools import Extension
import numpy as np
import os

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON_FLAG = True
except ImportError:
    CYTHON_FLAG = False

from setuptools.command import sdist


def extract_extension_list(file_format):
    extension_list = []
    for dirpath, dirs, files in os.walk('.'):
        for f in files:
            if (file_format in f and "checkpoint" not in f) or "__init__.py" == f:
                pyx_path = os.path.join(dirpath, f)
                print("Extract " + str(pyx_path))
                extension_list.append(Extension(pyx_path.replace("/", "."), [pyx_path]))

    return extension_list

def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


if CYTHON_FLAG is True:
    file_format = ".pyx"
    cmdclass = {"build_ext": build_ext}
else:
    file_format = ".c"
    cmdclass = {}

ext_modules = extract_extension_list(file_format)

class Sdist(sdist.sdist):

    def __init__(self, *args, **kwargs):
        assert CYTHON_FLAG

        from Cython.Build import cythonize

        for e in ext_modules:
            for src in e.sources:
                if file_format in src:
                    print("cythonize " + str(src))
                    cythonize(src)

        super(sdist.sdist, self).__init__(*args, **kwargs)

cmdclass.setdefault("sdist", Sdist)
cmdclass["sdist"] = Sdist


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
    include_package_data=True,
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules, exclude=['**/__init__.py'])
)
