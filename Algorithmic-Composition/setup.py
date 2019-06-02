# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_readme(file_name):
    from os import path
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, file_name), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(
    name='pycomposer',
    version='1.0.2',
    description='pycomposer is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory and the Statistical machine learning problems. Especialy, this library provides apprication of the Algorithmic Composer based on Generative Adversarial Networks(GANs) and Conditional Generative Adversarial Networks(Conditional GANs).',
    long_description=read_readme("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/chimera0/accel-brain-code/tree/master/Algorithmic-Composition',
    author='chimera0',
    author_email='ai-brain-lab@accel-brain.com',
    license='GPL2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='GAN GANs MIDI Composition Generative Adversarial Networks Conditional MidiNet MuseGAN',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas', 'pretty_midi', 'pygan', 'pydbm'],
)
