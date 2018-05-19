# Accel Brain Code: From Proof of Concept to Prototype.

The purpose of this repository is to make prototypes as case study in the context of proof of concept(PoC) that I have written in my website: [Accel Brain](https://accel-brain.com) (Japanese). Especially, natural language processing, statistical machine learning, and deep reinforcement learning are main topics.

## Description

The basic theme in my PoC is a lifeHack, which is any technique that reduces the burden of our life and make it easier to control, or more convenient. Considering that many lifehack solutions are technological and obviously product design and development technology are kind of *life* which can be *hacked*, this lifehack itself also can be purpose of lifehack. Because of this *Autologie*, a seemingly endless round of my PoC and technological prototypes is rotary driven by *Selbstreferenz* to repeat *lifehack of lifehack* cyclically.

In this problem setting and *recursive* solutions, this repository is functionally differentiated by compositions such as information collection, searching optimal solution, and focus booster. Each function can be considered an integral component of lifehack solutions. All code is implemented as an algorithm of machine learning or data science. Especially, natural language processing, statistical machine learning, and deep reinforcement learning are main topics.

# Prototypes

## [Automatic-Summarization](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization)

`pysummarization` is Python3 library for the automatic summarization, document abstraction, and text filtering.

The function of this library is automatic summarization using a kind of natural language processing. This library enable you to create a summary with the major points of the original document or web-scraped text that filtered by text clustering.

### Documentation

Full documentation is available on [https://code.accel-brain.com/Automatic-Summarization/](https://code.accel-brain.com/Automatic-Summarization/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## [Deep-Learning-by-means-of-Design-Pattern](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern)

`pydbm` is Python library for building Restricted Boltzmann Machine(RBM), Deep Boltzmann Machine(DBM), Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent neural network Restricted Boltzmann Machine(RNN-RBM), and Shape Boltzmann Machine(Shape-BM). This is **Cython version**. [pydbm_mxnet](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern/mxnet) (MXNet version) is derived from this library.

The function of this library is building and modeling Restricted Boltzmann Machine(RBM) and Deep Boltzmann Machine(DBM). The models are functionally equivalent to stacked auto-encoder. The basic function is the same as dimensions reduction(or pre-training). And this library enables you to build many functional extensions from RBM and DBM such as Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent Neural Network Restricted Boltzmann Machine(RNN-RBM), and Shape Boltzmann Machine(Shape-BM).

As more usecases, RTRBM and RNN-RBM can learn dependency structures in temporal patterns such as music, natural sentences, and n-gram. RTRBM is a probabilistic time-series model which can be viewed as a temporal stack of RBMs, where each RBM has a contextual hidden state that is received from the previous RBM and is used to modulate its hidden units bias. The RTRBM can be understood as a sequence of conditional RBMs whose parameters are the output of a deterministic RNN, with the constraint that the hidden units must describe the conditional distributions. This constraint can be lifted by combining a full RNN with distinct hidden units. In terms of this possibility, RNN-RBM is structurally expanded model from RTRBM that allows more freedom to describe the temporal dependencies involved.

On the other hand, the usecases of Shape-BM are image segmentation, object detection, inpainting and graphics. Shape-BM is the model for the task of modeling binary shape images, in that samples from the model look realistic and it can generalize to generate samples that differ from training examples.

<table border="0">
    <tr>
        <td>
            <img src="https://github.com/chimera0/accel-brain-code/raw/master/Deep-Learning-by-means-of-Design-Pattern/img/horse099.jpg" />
        <p>Image in <a href="https://avaminzhang.wordpress.com/2012/12/07/%E3%80%90dataset%E3%80%91weizmann-horses/" target="_blank">the Weizmann horse dataset</a>.</p>
        </td>
        <td>
            <img src="https://github.com/chimera0/accel-brain-code/raw/master/Deep-Learning-by-means-of-Design-Pattern/img/horse099_binary.png" />
            <p>Binarized image.</p>
        </td>
        <td>
            <img src="https://github.com/chimera0/accel-brain-code/raw/master/Deep-Learning-by-means-of-Design-Pattern/img/reconstructed_horse099.gif" />
            <p>Reconstructed image by Shape-BM.</p>
        </td>
    </tr>
</table>

In relation to my [Automatic Summarization Library](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization), it is important for me that the models are functionally equivalent to stacked auto-encoder. The main function I observe is the same as dimensions reduction(or pre-training). But the functional reusability of the models can be not limited to this. These Python Scripts can be considered a kind of experiment result to verify effectiveness of object-oriented analysis, object-oriented design, and GoF's design pattern in designing and modeling neural network, deep learning, and [Reinforcement-Learning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning).

### Documentation

Full documentation is available on [https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/](https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## [Reinforcement-Learning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning)

`pyqlearning` is Python library to implement Reinforcement Learning, especially for Q-Learning.

Considering many variable parts and functional extensions in the Q-learning paradigm, I implemented these Python Scripts for demonstrations of commonality/variability analysis in order to design the models.

I have details of this library on my Jupyter notebook: [search_maze_by_q_learning.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/search_maze_by_q_learning.ipynb). This notebook demonstrates a simple maze solving algorithm based on Epsilon-Greedy Q-Learning or Q-Learning, loosely coupled with Deep Boltzmann Machine(DBM) which can be implemented by importing `pydbm`.

As shown in the below image, the state-action value function and parameters setting can be designed to correspond with the optimality of route.

<div align="center">
 <table style="border: none;">
  <tr>
   <td width="45%" align="center">
    <a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/search_maze_by_q_learning.ipynb" target="_top"><img src="https://github.com/chimera0/accel-brain-code/raw/master/Reinforcement-Learning/img/maze_map.png" /></a>
    <p>Maze map</p>
   </td>
   <td width="45%" align="center">
    <a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/search_maze_by_q_learning.ipynb" target="_top"><img src="https://github.com/chimera0/accel-brain-code/raw/master/Reinforcement-Learning/img/feature_point.png" /></a>
    <p>Feature Points in the maze map</p>
   </td>
  </tr>
  <tr>
   <td width="45%" align="center">
    <a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/search_maze_by_q_learning.ipynb" target="_top"><img src="https://github.com/chimera0/accel-brain-code/raw/master/Reinforcement-Learning/img/fail_searched.png" /></a>
    <p>The result of searching by Epsilon-Greedy Q-Learning</p>
   </td>
   <td width="45%" align="center">
    <a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/search_maze_by_q_learning.ipynb" target="_top"><img src="https://github.com/chimera0/accel-brain-code/raw/master/Reinforcement-Learning/img/maze_q_learning_result.png"  /></a>
    <p>The result of searching by Q-Learning, loosely coupled with Deep Boltzmann Machine.</p>
   </td>
  </tr>
 </table>
</div>

### Documentation

Full documentation is available on [https://code.accel-brain.com/Reinforcement-Learning/](https://code.accel-brain.com/Reinforcement-Learning/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## [Reinforcement-Learning-with-js](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning-with-js)

These JavaScript modules are library to implement Reinforcement Learning, especially for Q-Learning. These modules are functionally equivalent to [pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning).

## [Algorithmic-Composition](https://github.com/chimera0/accel-brain-code/tree/master/Algorithmic-Composition)

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition by Reinforcement Learning such as Q-Learning and Recurrent Temporal Restricted Boltzmann Machine(RTRBM). Q-Learning and RTRBM in this library allows you to extract the melody information about a MIDI tracks and these models can learn and inference patterns of the melody. And This library has wrapper class for converting melody data inferenced by Q-Learning and RTRBM into MIDI file.

### Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic-Composition/](https://code.accel-brain.com/Algorithmic-Composition/). This document contains information on functionally reusability, functional scalability and functional extensibility.

## [Cardbox](https://github.com/chimera0/accel-brain-code/tree/master/Cardbox)

This is the simple card box system that make you able to find and save your ideas.

You can write down as many ideas as possible onto cards. Like the KJ Method or the mindmap tools, this simple JavaScript tool helps us to discover potential relations among the cards that you created. And the tagging function allow you to generate metadata of cards as to make their meaning and relationships understandable.

## [Binaural-Beat-and-Monaural-Beat-with-python](https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-python)

`AccelBrainBeat` is a Python library for creating the binaural beats or monaural beats. You can play these beats and generate wav files. The frequencys can be optionally selected.

This Python script enables you to handle your mind state by a kind of "Brain-Wave Controller" which is generally known as Biaural beat or Monauarl beats in a simplified method.

### Documentation

Full documentation is available on [https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/](https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## [Binaural-Beat-and-Monaural-Beat-with-js](https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-js)

These modules are functionally equivalent to Python Scripts in `AccelBrainBeat`.

## [Subliminal perception](https://github.com/chimera0/accel-brain-code/tree/master/Subliminal-Perception)

These JavaScript are tool for experimentation of subliminal perception.

This is a demo code for my case study in the context of my website.

# Author

- chimera0(RUM)

# Author URI

- http://accel-brain.com/

# License

- GNU General Public License v2.0
