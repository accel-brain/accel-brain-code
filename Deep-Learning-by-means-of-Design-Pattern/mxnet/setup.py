# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='pydbm_mxnet',
    version='1.0.1',
    description='pydbm-mxnet is Python library based on MXNet for building restricted boltzmann machine, deep boltzmann machine, and multi-layer neural networks.',
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
    keywords='restricted boltzmann machine autoencoder auto-encoder MXNet',
    install_requires=['numpy', 'mxnet'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*'])
)
