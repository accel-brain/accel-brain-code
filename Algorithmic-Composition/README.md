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
- [pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning)
- [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern)

## Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic-Composition/](https://code.accel-brain.com/Algorithmic-Composition/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Description

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory. Especialy, this library provides apprication of the generative model such as a **Restricted Boltzmann Machine**(RBM), which can be expanded as a **Recurrent Temporal Restricted Boltzmann Machine**(RTRBM) to learn probability distribution of tone row, pitch classes, or time-series pattern of sounds. The function of RTRBM model is inferencing a linear succession of musical tones that the listener perceives as a single entity.

And the **Monte Carlo method** such as **Quantum Annealing** model, which can be considered as structural expansion of the **Simulated Annealing**, is used in this library as optimizer of compositions. Simulated Annealing is a probabilistic single solution based search method inspired by the annealing process in metallurgy. Annealing is a physical process referred to as tempering certain alloys of metal, glass, or crystal by heating above its melting point, holding its temperature, and then cooling it very slowly until it solidifies into a perfect crystalline structure. The simulation of this process is known as simulated annealing.

There are many functional extensions and functional equivalents of **Simulated Annealing**. For instance, **Adaptive Simulated Annealing**, also known as the very fast simulated reannealing, is a very efficient version of simulated annealing. And **Quantum Monte Carlo**, which is generally known a stochastic method to solve the Schrödinger equation, is one of the earliest types of solution in order to simulate the **Quantum Annealing** in classical computer. In summary, one of the function of this algorithm is to solve the ground state search problem which is known as logically equivalent to combinatorial optimization problem.

`pycomposer` is Python library which provides wrapper classes for extracting sequencial data from MIDI files, feature extraction from this sequencial data by generative models, generating new sequencial data by drawing random samples from a Gaussian distribution, optimizing generated data by anneling models, and converting optimized data into new MIDI file.

## Demonstration

Import Python modules.

```python
import numpy as np
from pycomposer.composition_controler import CompositionControler
from pycomposer.inferablepitch.rtrbm_inferer import RTRBMInferer
from pycomposer.midi_vectorlizer import MidiVectorlizer
```

Extract and convert MIDI file into `pd.DataFrame`.

```python
midi_vectorlizer = MidiVectorlizer()
tone_df = midi_vectorlizer.extract("path/to/your/midi/file.mid")
```

Instantiate Q-Learning and setup hyper-parameters. `r_dict` is the dict of reward values. The key of `r_dict` is to the value what the frequency ratio is to Q-Value.

```python
# Reward value.
r_dict = {
    (1, 1): 5.0,
    (15, 16): 0.0,
    (8, 9): 0.0,
    (5, 6): 0.5,
    (4, 5): 0.5,
    (2, 1): 5.0,
    (32, 45): 0.0,
    (2, 3): 5.0,
    (5, 8): 0.5,
    (3, 5): 0.5,
    (9, 16): 0.0,
    (8, 15): 0.0,
    (1, 2): 5.0
}

# The object of Q-Learning.
inferable_consonance = QConsonance()
# ε-Greedy rate.
inferable_consonance.epsilon_greedy_rate = 0.9
# Alpha value.
inferable_consonance.alpha_value = 0.6
# Gamma value.
inferable_consonance.gamma_value = 0.7
# Initialize.
inferable_consonance.initialize(r_dict=r_dict, tone_df=tone_df)
# Searching and learning.
for pitch in tone_df.pitch.values.tolist():
    # The parameter of `limit` is Number of learning.
    inferable_consonance.learn(state_key=pitch, limit=200)
```

Instantiate RTRBM and setup hyper-parameters.

```python
inferable_pitch = RTRBMInferer(
    learning_rate=0.00001,         # Learning rate.
    hidden_n=100,                  # The number of units in hidden layer.
    hidden_binary_flag=True,       # The activation is binary or not.
    inferancing_training_count=1,  # Training count in inferancing.
    r_batch_size=200               # The batch size in inferancing.
)
inferable_pitch.learn(
    tone_df=tone_df,               # Learned data.
    training_count=1,              # Training count in learning.
    batch_size=200                 # The batch size in learning.
)
```

Instantiate `CompositionControler` as follow.

```python
# The conctorler of `inferable_consonance` and `inferable_pitch`
composition_controler = CompositionControler(
    resolution=960,
    initial_tempo=120
)
```

Create chord progression. The parameter of `octave` is octave (-1 - 9). The options of `first_chord` are `I`, `II`, `III`, `IV`, `V`, `VI`, and `VII`. `total_measure_n` is the length of measure.

```python
chord_list = composition_controler.create_chord_list(
    octave=5, 
    first_chord="IV",
    total_measure_n=40
)
```

Compose chord progression.

```python
composition_controler.compose_chord(
    chord_list,                      # The list of `np.ndarray` that contains the string of Diatonic code.
    metronome_time=100,              # Metronome time.
    start_measure_n=0,               # The timing of the beginning of the measure.
    measure_n=8,                     # The number of measures.
    beat_n=8,                        # The number of beats.
    chord_instrument_num=34,         # MIDI program number (instrument index), in [0, 127].
    chord_velocity_range=(90, 95)    # The tuple of chord velocity in MIDI.
)
```

And compose melody.

```python
composition_controler.compose_melody(
    inferable_pitch,
    inferable_consonance,
    chord_list,
    total_measure_n=40,
    measure_n=8,
    start_measure_n=0,
    beat_n=8,
    metronome_time=100,
    melody_instrument_num=0,            # MIDI program number (instrument index), in [0, 127].
    melody_velocity_range=(120, 127)    # The tuple of melody velocity in MIDI.
)
```

The form of tuple `chord_velocity_range` and `melody_velocity_range` is (low velocity, high veloicty). The value of velocity is determined by `np.random.randint`.

Finally, convert these inferenced data into MIDI file.

```python
composition_controler.save("your/composed/midi/file.mid")
```


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
