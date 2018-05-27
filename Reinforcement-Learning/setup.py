# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def read_rst(file_name):
    from os import path
    with open(path.join(path.dirname(__file__), file_name)) as f:
        rst = f.read()
    return rst


setup(
    name='pyqlearning',
    version='1.0.9',
    description='pyqlearning is Python library to implement Reinforcement Learning, especially for Q-Learning.',
    long_description=read_rst("README.rst"),
    url='https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning',
    author='chimera0',
    author_email='ai-brain-lab@accel-brain.com',
    license='GPL2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Framework :: Robot Framework',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
    keywords='Q-Learning Deep Q-Network DQN DBM Reinforcement Learning Boltzmann',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas'],
)
