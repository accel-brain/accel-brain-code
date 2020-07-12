# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(
    name='accel-brain-base',
    version='1.0.0',
    description='accel-brain-base is a basic library of the Deep Learning for rapid development at low cost. This library makes it possible to design and implement deep learning, which must be configured as a complex system, by combining a plurality of functionally differentiated modules such as a Restricted Boltzmann Machine(RBM), Deep Boltzmann Machines(DBMs), a Stacked-Auto-Encoder, an Encoder/Decoder based on Long Short-Term Memory(LSTM), and a Convolutional Auto-Encoder(CAE).',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base/',
    author='accel-brain',
    author_email='info@accel-brain.co.jp',
    license='GPL2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Framework :: Robot Framework',
        'Topic :: Text Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='Generative Adversarial Networks Adversarial Auto-Encoders autoencoder auto-encoder convolution deconvolution encoder decoder LSTM Deep Q-Learning Deep Reinforcement Learning Deep Boltzmann Machines Restricted Boltzmann Machine Ladder Networks semi-supervised learning DRCN',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas'],
)
