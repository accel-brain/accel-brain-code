# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='pycomposer',
    version='0.0.2',
    description='pycomposer is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory. Especialy, this library provides apprication of the generative model such as a Restricted Boltzmann Machine(RBM). And the Monte Carlo method such as Quantum Annealing model is used in this library as optimizer of compositions.',
    long_description='pycomposer is Python library for Algorithmic Composition or Automatic Composition by Reinforcement Learning such as Q-Learning and Recurrent Temporal Restricted Boltzmann Machine(RTRBM). Q-Learning and RTRBM in this library allows you to extract the melody information about a MIDI tracks and these models can learn and inference patterns of the melody. And This library has wrapper class for converting melody data inferenced by Q-Learning and RTRBM into MIDI file.',
    url='https://github.com/chimera0/accel-brain-code/tree/master/Algorithmic-Composition',
    author='chimera0',
    author_email='ai-brain-lab@accel-brain.com',
    license='GPL2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='RTRBM LSTM Annealing Quantum Monte Carlo',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas', 'pretty_midi', 'pydbm', 'pyqlearning'],
)
