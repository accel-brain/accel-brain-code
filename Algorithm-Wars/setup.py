# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description

setup(
    name='algo-wars',
    version='0.0.1',
    description='algo-wars is Python library for Investment Strategies such as Volatility Modeling, Technical Analysis, and Portfolio Optimization. The goal of this library is to provides cues for strategies to investment or trade stock, bond, or cryptocurrency, based on the statistical machine learning and the deep reignforcement learning.',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/accel-brain/accel-brain-code/tree/master/Algorithm-Wars/',
    author='accel-brain',
    author_email='info@accel-brain.co.jp',
    license='GPL2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'accel-brain-base',
        'pandas',
        'numpy'
    ]
)
