# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_rst(file_name):
    from os import path
    with open(path.join(path.dirname(__file__), file_name)) as f:
        rst = f.read()
    return rst

setup(
    name='AccelBrainBeat',
    version='1.0.5',
    description='AccelBrainBeat is a Python library for creating the binaural beats or monaural beats. You can play these beats and generate wav files. The frequencys can be optionally selected.',
    long_description=read_rst("README.rst"),
    url='https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-python',
    author='chimera0',
    author_email='ai-brain-lab@accel-brain.com',
    license='GPL2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
    ],
    keywords='binaural monaural beats brain wave wav audio',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy'],
)
