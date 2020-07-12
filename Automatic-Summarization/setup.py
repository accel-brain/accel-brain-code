# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description

setup(
    name='pysummarization',
    version='1.1.5',
    description='pysummarization is Python library for the automatic summarization, document abstraction, and text filtering in relation to Encoder/Decoder based on LSTM and LSTM-RTRBM.',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/accel-brain/accel-brain-code/tree/master/Automatic-Summarization/',
    author='accel-brain',
    author_email='info@accel-brain.co.jp',
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
    keywords='Automatic summarization document abstraction abstract text filtering',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'nltk']
)
