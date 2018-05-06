# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_rst(file_name):
    from os import path
    with open(path.join(path.dirname(__file__), file_name)) as f:
        rst = f.read()
    return rst


setup(
    name='pysummarization',
    version='1.0.6',
    description='pysummarization is Python library for the automatic summarization, document abstraction, and text filtering.',
    long_description=read_rst("README.rst"),
    url='https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization',
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
    keywords='Automatic summarization document abstraction abstract text filtering',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'nltk', 'mecab-python3', 'pyquery', 'pdfminer2'],
)
