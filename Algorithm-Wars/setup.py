# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='algo-wars',
    version='0.0.1',
    description='algo-wars.',
    long_description='algo-wars.',
    #long_description_content_type='text/markdown',
    url='https://github.com/accel-brain/accel-brain-code/tree/master/Algorithm-Wars/',
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
    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'accel-brain-base',
    ]
)
