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
    version='0.0.2',
    description='.',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base/',
    author='accel-brain',
    author_email='info@accel-brain.co.jp',
    license='GPL2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Framework :: Robot Framework',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='Generative Adversarial Networks Adversarial Auto-Encoders autoencoder auto-encoder convolution deconvolution encoder decoder LSTM Deep Q-Learning Deep Reinforcement Learning Deep Boltzmann Machines Restricted Boltzmann Machine Ladder Networks semi-supervised learning DRCN',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas'],
)
