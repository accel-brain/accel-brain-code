# Algorithmic Composition or Automatic Composition Library: pycomposer

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition by Reinforcement Learning such as Q-Learning and Recurrent Temporal Restricted Boltzmann Machine(RTRBM).

This is BETA version.

## Description

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition by Reinforcement Learning such as Q-Learning and Recurrent Temporal Restricted Boltzmann Machine(RTRBM). Q-Learning and RTRBM in this library allows you to extract the melody information about a MIDI tracks and these models can learn and inference patterns of the melody. And This library has wrapper class for converting melody data inferenced by Q-Learning and RTRBM into MIDI file.

### Reinforcement Learning such as Q-Learning

Q-Learning is a kind of `Temporal Difference learning`(`TD Learning`) that can be considered as hybrid of `Monte Carlo method` and `Dynamic Programming Method`. As `Monte Carlo method`, `TD Learning` algorithm can learn by experience without model of environment. And this learning algorithm is *functionally equivalent* of bootstrap method as `Dynamic Programming Method`.

`Epsilon Greedy Q-Leanring` algorithm is `off-policy`. In this paradigm, *stochastic* searching and *deterministic* searching can coexist by hyperparameter ε (0 < ε < 1) that is probability that agent searches greedy. Greedy searching is *deterministic* in the sense that policy of agent follows the selection that maximizes the Q-Value.

In this library, this Q-Learning algorithm is implemented based on my [pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning) that can offer a broad range of concrete functions. For instance, adjustment and tuning of temperament or so-called Musical temperament such as Equal temperament, Meantone temperament, and 12 temperament can be easy-to-follow example as usecase. These temperament is the provision of the relative relationship of the pitches used for music. Although pure temperament frequency ratio corresponds to the ratio of whole numbers and then we can *discretely* distinguish between **Consonance** and **Dissonance**, but considering many variable parts, in engineering, that ratio should be measured as *continuous* and quantitative degree of consonance.

In relation to reinforcement learning theory, this degree of consonance can be considered as Q-Value. The state-action value function `Q(state, action)` can be defined as `Q(last sound,  current sound)` that returns the degree of consonance of `last sound` as state and `current sound` as action. In other words, this Q-Learning `agent` searches for the Combinatorial Optimization of two sounds. However, more strictly, the main function of Q-Learning in this library is not selecting only optimal solution but scoring the degrees based on the relative evaluation.

### Recurrent Temporal Restricted Boltzmann Machine(RTRBM)

As illustrated in my [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern), Recurrent Temporal Restricted Boltzmann Machine(RTRBM) is a probabilistic time-series model which can be viewed as a temporal stack of RBMs, where each RBM has a contextual hidden state that is received from the previous RBM and is used to modulate its hidden units bias. Then this model can learn dependency structures in temporal patterns such as music, natural sentences, and n-gram.

As concrete usecase of Algorithmic Composition, RTRBM can be considered as generative model to learn probability distribution of tone row, pitch classes, or time-series pattern of sounds. The function of RTRBM model is inferencing a linear succession of musical tones that the listener perceives as a single entity.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic Composition/](https://code.accel-brain.com/Algorithmic Composition/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

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
