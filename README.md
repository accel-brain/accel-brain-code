# Accel Brain Code: From Proof of Concept to Prototype.

The purpose of this repository is to make prototypes as case study in the context of proof of concept(PoC) and research and development(R&D) that I have written in my website: [Accel Brain](https://accel-brain.com) (Japanese) and [Accel Brain Co., Ltd.](https://accel-brain.co.jp/) (Japanese). The main research topics are Auto-Encoders in relation to the representation learning, the statistical machine learning for energy-based models, adversarial generation networks(GANs), Deep Reinforcement Learning such as Deep Q-Networks, semi-supervised learning, and neural network language model for natural language processing.

## Problem Setting: Deep Learning after the era of "Democratization of Artificial Intelligence(AI)".

How the Research and Development(R&D) on the subject of machine learning including deep learning, after the era of "Democratization of Artificial Intelligence(AI)", can become possible? Simply implementing the models and algorithms provided by standard machine learning libraries and applications like AutoML would reinvent the wheel. If you just copy and paste the demo code from the library and use it, your R&D would fall into dogmatically authoritarian development, or so-called the Hype driven development.

If you fall in love with the concept of "Democratization of AI," you may forget the reality that the R&D is under the influence of not only democracy but also capitalism. The R&D provides economic value when its R&D artifacts are distinguished from the models and algorithms realized by standard machine learning libraries and applications such as AutoML. In general terms, R&D must provide a differentiator to maximize the scarcity of its implementation artifacts.

On the other hand, it must be remembered that any R&D builds on the history of the social structure and the semantics of the concepts envisioned by previous studies. Many models and algorithms are variants derived not only from research but also from the relationship with business domains. It is impossible to assume differentiating factors without taking commonality and identity between society and its history.

### Problem Solution: PoC of PoC.

The blind spot of "democratization of AI" occurs when a new concept is created throughout the society, including business. It takes time before a new concept can be broken down into an interface specification from a perspective such as object-oriented analysis, and code that conforms to the interface specification can be implemented. There will always be some difference between the new AI created in this way and the AI already "democratized". 

In a more realistic perspective, casual users who are just waiting for the AI to be "democratized" will always fall behind. On the contrary, those who can create new concepts and new AIs with PoC will always continue to have a leading advantage in the market where AI is the main topic. Hiding behind the "democratic" movement of "AI democratization" is the dry reality of "capitalist" competition.

#### Lifehack of Lifehack.

The basic theme in my PoC is a Lifehack, which is any technique that reduces the burden of our life and make it easier to control, or more convenient. Considering that many lifehack solutions are technological and obviously product design and development technology are kind of *life* which can be *hacked*, lifehack itself also can be purpose of lifehack. Because of this *Autologie*, a seemingly endless round of my PoC and technological prototypes is rotary driven by *Selbstreferenz* to repeat *lifehack of lifehack* cyclically.

In this problem setting and *recursive* solutions, this repository is functionally differentiated by compositions such as information collection, searching optimal solution, and focus booster. Each function can be considered an integral component of lifehack solutions. These tools make it possible to efficiency the process of contemplation and accelerate our brain, enabling provisions for the developments of other tools in this repository. All code, implemented as in an algorithm of machine learning or data science, reflects the concept of proof of concept(PoC). 

### Problem Solution: [Accel-Brain-Base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).

As part of prototyping, this repository publishes a special machine learning library, [Accel-Brain-Base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).

Considering many variable parts, structural unions, and *functional equivalents* in the deep learning paradigm, which are variants derived not only from research but also from the relationship with business domains, from perspective of *commonality/variability analysis* in order to practice object-oriented design, this library provides abstract classes that define the skeleton of the deep Learning algorithm in an operation, deferring some steps in concrete variant algorithms such as the **Deep Boltzmann Machines**, **Stacked Auto-Encoder**, **Encoder/Decoder based on LSTM**, and **Convolutional Auto-Encoder** to client subclasses. The abstract classes and the interfaces in this library let subclasses redefine certain steps of the deep Learning algorithm without changing the algorithm's structure.

These abstract classes can also provide new original models and algorithms such as **Generative Adversarial Networks(GANs)**, **Deep Reinforcement Learning**, or **Neural network language model** by implementing the variable parts of the fluid elements of objects.

### Problem Solution: [Automatic Summarization Library: pysummarization](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization)

`pysummarization` is Python3 library for the automatic summarization, document abstraction, and text filtering.

The function of this library is automatic summarization using a kind of natural language processing. This library enable you to create a summary with the major points of the original document or web-scraped text that filtered by text clustering.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Automatic-Summarization/](https://code.accel-brain.com/Automatic-Summarization/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Deep Learning Library: pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern)

The function of `pydbm` is building and modeling **Restricted Boltzmann Machine**(RBM) and **Deep Boltzmann Machine**(DBM). The models are functionally equivalent to **stacked auto-encoder**. The basic function is the same as **dimensions reduction**(or **pre-training**). And this library enables you to build many functional extensions from RBM and DBM such as Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent Neural Network Restricted Boltzmann Machine(RNN-RBM), Long Short-Term Memory Recurrent Temporal Restricted Boltzmann Machine(LSTM-RTRBM), and Shape Boltzmann Machine(Shape-BM).

#### RTRBM, RNN-RBM, and LSTM-RTRBM for probabilistic time-series model.

As more usecases, **RTRBM**, **RNN-RBM**, and **LSTM-RTRBM** can learn dependency structures in temporal patterns such as music, natural sentences, and n-gram. RTRBM is a probabilistic time-series model which can be viewed as a temporal stack of RBMs, where each RBM has a contextual hidden state that is received from the previous RBM and is used to modulate its hidden units bias. The RTRBM can be understood as a sequence of conditional RBMs whose parameters are the output of a deterministic RNN, with the constraint that the hidden units must describe the conditional distributions. This constraint can be lifted by combining a full RNN with distinct hidden units. In terms of this possibility, RNN-RBM and LSTM-RTRBM are structurally expanded model from RTRBM that allows more freedom to describe the temporal dependencies involved.

#### Shape-BM for image segmentation, object detection, inpainting and graphics.

The usecases of **Shape-BM** are image segmentation, object detection, inpainting and graphics. Shape-BM is the model for the task of modeling binary shape images, in that samples from the model look realistic and it can generalize to generate samples that differ from training examples.

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

#### Functionally equivalent: Encoder/Decoder based on LSTM.

The methodology of *equivalent-functionalism* enables us to introduce more functional equivalents and compare problem solutions structured with different algorithms and models in common problem setting. For example, in dimension reduction problem, the function of **Encoder/Decoder schema** is equivalent to **DBM** as a **Stacked Auto-Encoder**.

<div><img src="https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/img/latex/encoder_decoder.png" />
<p>Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078., p2.</p>
</div>

According to the neural networks theory, and in relation to manifold hypothesis, it is well known that multilayer neural networks can learn features of observed data points and have the feature points in hidden layer. High-dimensional data can be converted to low-dimensional codes by training the model such as **Stacked Auto-Encoder** and **Encoder/Decoder** with a small central layer to reconstruct high-dimensional input vectors. This function of dimensionality reduction facilitates feature expressions to calculate similarity of each data point.

This library provides **Encoder/Decoder based on LSTM**, which is a reconstruction model and makes it possible to extract series features embedded in deeper layers. The LSTM encoder learns a fixed length vector of time-series observed data points and the LSTM decoder uses this representation to reconstruct the time-series using the current hidden state and the value inferenced at the previous time-step.

#### Functionally equivalent: Convolutional Auto-Encoder.

**Shape-BM** is a kind of problem solution in relation to problem settings such as image segmentation, object detection, inpainting and graphics. In this problem settings, **Convolutional Auto-Encoder**(Masci, J., et al., 2011) is a functionally equivalent of **Shape-BM**. A stack of Convolutional Auto-Encoder forms a convolutional neural network(CNN), which are among the most successful models for supervised image classification. Each Convolutional Auto-Encoder is trained using conventional on-line gradient descent without additional regularization terms.

<table border="0">
    <tr>
        <td>
            <img src="https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/img/horse099.jpg" />
        <p>Image in <a href="https://avaminzhang.wordpress.com/2012/12/07/%E3%80%90dataset%E3%80%91weizmann-horses/" target="_blank">the Weizmann horse dataset</a>.</p>
        </td>
        <td>
            <img src="https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/img/reconstructed_horse099.gif" />
            <p>Reconstructed image by <strong>Shape-BM</strong>.</p>
        </td>
        <td>
            <img src="https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/img/reconstructed_by_CAE.gif" />
            <p>Reconstructed image by <strong>Convolutional Auto-Encoder</strong>.</p>
        </td>
    </tr>
</table>

My jupyter notebook: [demo/demo_convolutional_auto_encoder.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_convolutional_auto_encoder.ipynb) also demonstrates various reconstructed images.

In this library, **Convolutional Auto-Encoder** is also based on **Encoder/Decoder** scheme. The *encoder* is to the *decoder* what the *Convolution* is to the *Deconvolution*. The Deconvolution also called transposed convolutions "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

#### Structural expansion for Spatio-Temporal Auto-Encoder.

**Encoder/Decoder based on LSTM** and **Convolutional Auto-Encoder** have a functional reusability to extend the structures to **Spatio-Temporal Auto-Encoder**, which can learn the regular patterns in the training videos(Baccouche, M., et al., 2012, Patraucean, V., et al. 2015). This model consists of spatial Auto-Encoder and temporal Encoder/Decoder. The spatial Auto-Encoder is a Convolutional Auto-Encoder for learning spatial structures of each video frame. The temporal Encoder/Decoder is an Encoder/Decoder based on LSTM scheme for learning temporal patterns of the encoded spatial structures. The spatial encoder and decoder have two convolutional and deconvolutional layers respectively, while the temporal encoder and decoder are to act as a twin LSTM models.

<div><img src="https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/img/spatio_temporal_auto_encoder_model.png" />
<p>Chong, Y. S., & Tay, Y. H. (2017, June). Abnormal event detection in videos using spatiotemporal autoencoder. In International Symposium on Neural Networks (pp. 189-196). Springer, Cham., p.195.</p>
</div>

##### Video recognition and reconstruction of video images.

[demo/demo_spatio_temporal_auto_encoder.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_spatio_temporal_auto_encoder.ipynb) is a jupyter notebook which demonstrates the video recognition and reconstruction of video images by the Spatio-Temporal Auto-Encoder.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/](https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Reinforcement Learning Library: pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning)

`pyqlearning` is Python library to implement Reinforcement Learning and Deep Reinforcement Learning, especially for Q-Learning, Deep Q-Network, and Multi-agent Deep Q-Network which can be optimized by Annealing models such as Simulated Annealing, Adaptive Simulated Annealing, and Quantum Monte Carlo Method.

According to the Reinforcement Learning problem settings, Q-Learning is a kind of **Temporal Difference learning(TD Learning)** that can be considered as hybrid of **Monte Carlo** method and **Dynamic Programming** method. As Monte Carlo method, TD Learning algorithm can learn by experience without model of environment. And this learning algorithm is functional extension of bootstrap method as Dynamic Programming Method.

#### The commonality/variability of Q-Learning.

In this library, Q-Learning can be distinguished into **Epsilon Greedy Q-Leanring** and **Boltzmann Q-Learning**. These algorithm is functionally equivalent but their structures should be conceptually distinguished.

Considering many variable parts and functional extensions in the Q-learning paradigm from perspective of *commonality/variability* analysis in order to practice object-oriented design, this library provides abstract class that defines the skeleton of a Q-Learning algorithm in an operation, deferring some steps in concrete variant algorithms such as Epsilon Greedy Q-Leanring and Boltzmann Q-Learning to client subclasses. The abstract class in this library lets subclasses redefine certain steps of a Q-Learning algorithm without changing the algorithm's structure.

#### Simple Maze Solving by Deep Q-Network

[demo/search_maze_by_deep_q_network.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/demo/search_maze_by_deep_q_network.ipynb) is a Jupyter notebook which demonstrates a maze solving algorithm based on Deep Q-Network, rigidly coupled with Deep Convolutional Neural Networks(Deep CNNs). The function of the Deep Learning is **generalisation** and CNNs is-a **function approximator**. In this notebook, several functional equivalents such as CNN, Long Short-Term Memory(LSTM) networks, and the model which loosely coupled CNN and LSTM can be compared from a functional point of view.

<div align="center">
    <p><a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/demo/search_maze_by_deep_q_network.ipynb" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Reinforcement-Learning/img/DQN_single_agent_goal_compressed.gif" /></a></p>
    <p>Deep Reinforcement Learning to solve the Maze.</p>
</div>

* Black squares represent a wall.
* Light gray squares represent passages.
* A dark gray square represents a start point.
* A white squeare represents a goal point.

##### The pursuit-evasion game

Expanding the search problem of the maze makes it possible to describe the pursuit-evasion game that is a family of problems in mathematics and computer science in which one group attempts to track down members of another group in an environment.

This problem can be re-described as the multi-agent control problem, which involves decomposing the global system state into an image like representation with information encoded in separate channels. This reformulation allows us to use convolutional neural networks to efficiently extract important features from the image-like state.

[demo/search_maze_by_deep_q_network.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/demo/search_maze_by_deep_q_network.ipynb) also prototypes Multi Agent Deep Q-Network to solve the pursuit-evasion game based on the image-like state representation of the multi-agent.

<div align="center">
    <table style="border: none;">
        <tr>
            <td width="45%" align="center">
            <p><a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/demo/search_maze_by_deep_q_network.ipynb" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Reinforcement-Learning/img/DQN_multi_agent_demo_crash_enemy_2-compressed.gif" /></a></p>
            <p>Multi-agent Deep Reinforcement Learning to solve the pursuit-evasion game. The player is caught by enemies.</p>
            </td>
            <td width="45%" align="center">
            <p><a href="https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/demo/search_maze_by_deep_q_network.ipynb" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Reinforcement-Learning/img/DQN_multi_agent_demo_goal_enemy_2-compressed.gif" /></a></p>
            <p>
            <p>Multi-agent Deep Reinforcement Learning to solve the pursuit-evasion game. The player reaches the goal.</p>
            </td>
        </tr>
    </table>
</div>

* Black squares represent a wall.
* Light gray squares represent passages.
* A dark gray square represents a start point.
* Moving dark gray squares represent enemies.
* A white squeare represents a goal point.

#### Combinatorial optimization problem and Simulated Annealing.

There are many hyperparameters that we have to set before the actual searching and learning process begins. Each parameter should be decided in relation to Reinforcement Learning theory and it cause side effects in training model. This issue can be considered as **Combinatorial optimization problem** which is an optimization problem, where an optimal solution has to be identified from a finite set of solutions. In this problem setting, this library provides an Annealing Model such as **Simulated Annealing** to search optimal combination of hyperparameters. 

As exemplified in [annealing_hand_written_digits.ipynb](https://github.com/chimera0/accel-brain-code/blob/master/Reinforcement-Learning/annealing_hand_written_digits.ipynb), there are many functional extensions and functional equivalents of Simulated Annealing. For instance, **Adaptive Simulated Annealing**, also known as the very fast simulated reannealing, is a very efficient version of simulated annealing. And **Quantum Monte Carlo**, which is generally known a stochastic method to solve the Schrödinger equation, is one of the earliest types of solution in order to simulate the **Quantum Annealing** in classical computer.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Reinforcement-Learning/](https://code.accel-brain.com/Reinforcement-Learning/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Generative Adversarial Networks Library: pygan](https://github.com/chimera0/accel-brain-code/tree/master/Generative-Adversarial-Networks)

`pygan` is Python library to implement Generative Adversarial Networks(GANs) and Adversarial Auto-Encoders(AAEs).

This library makes it possible to design the Generative models based on the Statistical machine learning problems in relation to Generative Adversarial Networks(GANs) and Adversarial Auto-Encoders(AAEs) to practice algorithm design for semi-supervised learning.

The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes a
min-max adversarial game between two neural networks – a generative model, `G`, and a discriminative
model, `D`. The discriminator model, `D(x)`, is a neural network that computes the probability that
a observed data point `x` in data space is a sample from the data distribution (positive samples) that we are trying to model, rather than a sample from our generative model (negative samples). Concurrently, the generator uses a function `G(z)` that maps samples `z` from the prior `p(z)` to the data space. `G(z)` is trained to maximally confuse the discriminator into believing that samples it generates come from the data distribution. The generator is trained by leveraging the gradient of `D(x)` w.r.t. `x`, and using that to modify its parameters.

This library provides the Adversarial Auto-Encoders(AAEs), which is a probabilistic Auto-Encoder that uses GANs to perform variational inference by matching the aggregated posterior of the feature points in hidden layer of the Auto-Encoder with an arbitrary prior distribution(Makhzani, A., et al., 2015). Matching the aggregated posterior to the prior ensures that generating from any part of prior space results in meaningful samples. As a result, the decoder of the Adversarial Auto-Encoder learns a deep generative model that maps the imposed prior to the data distribution.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Generative-Adversarial-Networks/](https://code.accel-brain.com/Generative-Adversarial-Networks/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Algorithmic-Composition](https://github.com/chimera0/accel-brain-code/tree/master/Algorithmic-Composition)

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition by Reinforcement Learning such as Q-Learning and Recurrent Temporal Restricted Boltzmann Machine(RTRBM). Q-Learning and RTRBM in this library allows you to extract the melody information about a MIDI tracks and these models can learn and inference patterns of the melody. And This library has wrapper class for converting melody data inferenced by Q-Learning and RTRBM into MIDI file.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic-Composition/](https://code.accel-brain.com/Algorithmic-Composition/). This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Cardbox](https://github.com/chimera0/accel-brain-code/tree/master/Cardbox)

This is the simple card box system that make you able to find and save your ideas.

You can write down as many ideas as possible onto cards. Like the KJ Method or the mindmap tools, this simple JavaScript tool helps us to discover potential relations among the cards that you created. And the tagging function allow you to generate metadata of cards as to make their meaning and relationships understandable.

### Problem Solution: [Binaural-Beat-and-Monaural-Beat-with-python](https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-python)

`AccelBrainBeat` is a Python library for creating the binaural beats or monaural beats. You can play these beats and generate wav files. The frequencys can be optionally selected.

This Python script enables you to handle your mind state by a kind of "Brain-Wave Controller" which is generally known as Biaural beat or Monauarl beats in a simplified method.

#### Documentation

Full documentation is available on [https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/](https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

### Problem Solution: [Binaural-Beat-and-Monaural-Beat-with-js](https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-js)

These modules are functionally equivalent to Python Scripts in `AccelBrainBeat`.

### Problem Solution: [Subliminal perception](https://github.com/chimera0/accel-brain-code/tree/master/Subliminal-Perception)

These JavaScript are tool for experimentation of subliminal perception.

This is a demo code for my case study in the context of my website.

# Author

- Accel Brain Co., Ltd.

# Author URI

- http://accel-brain.com/
- http://accel-brain.co.jp/

# License

- GNU General Public License v2.0
