# Algorithmic Composition or Automatic Composition Library: pycomposer

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory. Especialy, this library provides apprication of the generative model such as a Restricted Boltzmann Machine(RBM). And the Monte Carlo method such as Quantum Annealing model is used in this library as optimizer of compositions.

This is BETA version.

## Installation

Install using pip:

```sh
pip install pycomposer
```

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Algorithmic Composition](https://github.com/chimera0/accel-brain-code/tree/master/Algorithmic-Composition)

### Python package index(PyPI)

Installers for the latest released version are available at the Python package index.

- [pycomposer : Python Package Index](https://pypi.org/pypi/pycomposer/)

### Dependencies

- numpy: v1.13.3 or higher.
- pandas: v0.22.0 or higher.
- pretty_midi: latest.
- [pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning): latest.
- [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern): latest.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic-Composition/](https://code.accel-brain.com/Algorithmic-Composition/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Description

`pycomposer` is Python library which provides wrapper classes for:

    - reading sequencial data from MIDI files, 
    - extracting feature points of observed data points from this sequencial data by *generative models*, 
    - generating new sequencial data by compositions based on the Twelve tone technique, 
    - optimizing generated data by *anneling models*, 
    - and converting optimized data into new MIDI file.

In order to realize these functions, this library implements two main algorithms: **Restricted Boltzmann Machine**(RBM) as a Generative model and **Quantum Monte Carlo**(QMC) as an Annealing model. The former model can inference feature points, which can be considered, if likened, prior knowledge and limiting conditions of composition. The function of the latter model is to minimize Kullback–Leibler divergences (KL divergences) of those inferenced feature points and new sequencial data points.

### Restricted Boltzmann Machine as a Generative model.

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory. Especialy, this library provides apprication of the generative model such as a **Restricted Boltzmann Machine**(RBM), which can be expanded as a **Recurrent Temporal Restricted Boltzmann Machine**(RTRBM) to learn probability distribution of tone row, pitch classes, or time-series pattern of sounds. The function of RTRBM model is inferencing a linear succession of musical tones that the listener perceives as a single entity.

The RTRBM can be understood as a sequence of conditional RBMs whose parameters are the output of a deterministic RNN, with the constraint that the hidden units must describe the conditional distributions. This constraint can be lifted by combining a full RNN with distinct hidden units. In terms of this possibility, **Recurrent Neural Network Restricted Boltzmann Machine**(RNN-RBM) and **Long Short-Term Memory Recurrent Temporal Restricted Boltzmann Machine**(LSTM-RTRBM) are structurally expanded model from RTRBM that allows more freedom to describe the temporal dependencies involved.

## Monte Carlo method as an Annealing model.

And the **Monte Carlo method** such as **Quantum Annealing** model, which can be considered as structural expansion of the **Simulated Annealing**, is used in this library as optimizer of compositions. Simulated Annealing is a probabilistic single solution based search method inspired by the annealing process in metallurgy. Annealing is a physical process referred to as tempering certain alloys of metal, glass, or crystal by heating above its melting point, holding its temperature, and then cooling it very slowly until it solidifies into a perfect crystalline structure. The simulation of this process is known as simulated annealing.

There are many structural extensions and functional equivalents of **Simulated Annealing**. For instance, **Adaptive Simulated Annealing**, also known as the very fast simulated reannealing, is a very efficient version of simulated annealing. And **Quantum Monte Carlo**, which is generally known a stochastic method to solve the Schrödinger equation, is one of the earliest types of solution in order to simulate the **Quantum Annealing** in classical computer. In summary, one of the function of this algorithm is to solve the ground state search problem which is known as logically equivalent to combinatorial optimization problem.

## Demonstration

Import Python modules.

```python
import numpy as np
from pycomposer.controller import Controller
```

Instantiate the controller object. This class wraps the function of LSTM-RTRBM and QMC. If `verbose` is `True`, this object prints transition status of optimization cost.

```python
controller = Controller(verbose=True)
```

Execute method `compose`.

```python
controller.compose(
    # The file path which is extracted in learning.
    learned_midi_path="/path/to/your/midi/input.mid",
    # Saved file path.
    saved_midi_path="/path/to/your/midi/output.mid",
    # One cycle length observed by RBM as one sequencial data.
    cycle_len=12,
    # The octave of music to be composed.
    octave=7,
    # Epoch in RBM's mini-batch training.
    epoch=1,
    # Batch size in RBM's mini-batch training.
    batch_size=50,
    # Learning rate for RBM.
    learning_rate=1e-05,
    # The number of units in hidden layer of RBM.
    hidden_num=100,
    # The number of cycles of annealing.
    annealing_cycles=100,
    # MIDI program number (instrument index), in [0, 127].
    program=0
)
```

Finally, new MIDI file is stored in `saved_midi_path`.

### Related PoC

- [量子力学、統計力学、熱力学における天才物理学者たちの神学的な形象について](https://accel-brain.com/das-theologische-bild-genialer-physiker-in-der-quantenmechanik-und-der-statistischen-mechanik-und-thermodynamik/) (Japanese)
    - [プロトタイプの開発：確率的音楽の統計力学的な自動作曲](https://accel-brain.com/das-theologische-bild-genialer-physiker-in-der-quantenmechanik-und-der-statistischen-mechanik-und-thermodynamik/5/#i-6)
- [深層強化学習のベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/) (Japanese)
    - [プロトタイプの開発：深層学習と強化学習による「排除された第三項」の推論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/4/#i-5)
- [ハッカー倫理に準拠した人工知能のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/) (Japanese)
    - [プロトタイプの開発：深層強化学習のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/5/#i-2)    

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
