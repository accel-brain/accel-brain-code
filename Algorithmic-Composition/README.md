# Algorithmic Composition or Automatic Composition Library: pycomposer

`pycomposer` is Python library for Algorithmic Composition or Automatic Composition based on the stochastic music theory and the Statistical machine learning problems. Especialy, this library provides apprication of the Algorithmic Composer based on Generative Adversarial Networks(GANs).

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
- pygan: latest.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Algorithmic-Composition/](https://code.accel-brain.com/Algorithmic-Composition/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Description

`pycomposer` is Python library which provides wrapper classes for:
- reading sequencial data from MIDI files, 
- extracting feature points of observed data points from this sequencial data by *generative models*, 
- generating new sequencial data by compositions based on Generative models, 
- and converting the data into new MIDI file.

### Generative Adversarial Networks(GANs)

In order to realize these functions, this library implements algorithms of Algorithmic Composer based on Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework which establishes a
min-max adversarial game between two neural networks – a generative model, `G`, and a discriminative
model, `D`. The discriminator model, `D(x)`, is a neural network that computes the probability that
a observed data point `x` in data space is a sample from the data distribution (positive samples) that we are trying to model, rather than a sample from our generative model (negative samples). Concurrently, the generator uses a function `G(z)` that maps samples `z` from the prior `p(z)` to the data space. `G(z)` is trained to maximally confuse the discriminator into believing that samples it generates come from the data distribution. The generator is trained by leveraging the gradient of `D(x)` w.r.t. `x`, and using that to modify its parameters.

### Architecture design

This library is designed following the above theory. The composer `GANComposer` learns observed data points drawn from a true distribution of input MIDI files and generates feature points drawn from a fake distribution that means such as Uniform distribution or Normal distribution, imitating the true MIDI files data.

The components included in this composer are functionally differentiated into three models.

1. `TrueSampler`.
2. `Generator`.
3. `Discriminator`.

The function of `TrueSampler` is to draw samples from a true distribution of input MIDI files.  `Generator` has `NoiseSampler`s and draw fake samples from a Uniform distribution or Normal distribution by use it. And `Discriminator` observes those input samples, trying discriminating true and fake data. 

By default, `Generator` and `Discriminator` are built as LSTM networks, observing MIDI data separated by fixed sequence length and time resolution. While `Discriminator` observes `Generator`'s observation to discrimine the output from true samples,  `Generator` observes `Discriminator`'s observations to confuse `Discriminator`s judgments. In GANs framework, the mini-max game can be configured by the observations of observations.

After this game, the `Generator` will grow into a functional equivalent that enables to imitate the `TrueSampler` and makes it possible to compose similar but slightly different music by the imitation.

## Demonstration

Import Python modules.

```python
from pycomposer.gan_composer import GANComposer
```

Instantiate the controller object.

```python
gan_composer = GANComposer(
    # `list` of paths to MIDI files.
    midi_path_list=[
        "path/to/your/midi/files.mid",
        "path/to/your/midi/files.mid",
        "path/to/your/midi/files.mid"
    ], 
    # Program in generated MIDI.
    target_program=0,
    # Batch size.
    batch_size=10,
    # The length of sequence that LSTM networks will observe.
    seq_len=4,
    # Time fraction or time resolution (seconds).
    time_fraction=0.5
)
```

Execute learning.

```python
gan_composer.learn(
    # The number of training iterations.
    iter_n=1000, 
    # The number of learning of the `discriminator`.
    k_step=10
)
```

After learning, `gan_composer` enables to compose melody and new MIDI file by learned model. In relation to MIDI data, `pitch` is generated from a learned generation model. `start` and `end` are generated by calculating back from length of sequences and time resolution. On the other hand, `velocity` is sampled from Gaussian distribution.

```python
gan_composer.compose(
    # Path to generated MIDI file.
    file_path="path/to/new/midi/file.mid", 
    # Minimum of pitch.
    # This class generates the pitch in the range 
    # `pitch_min` to `pitch_min` + 12.
    # If `None`, the average pitch in MIDI files set to this parameter.
    pitch_min=None,
    # Mean of velocity.
    # This class samples the velocity from a Gaussian distribution of 
    # `velocity_mean` and `velocity_std`.
    # If `None`, the average velocity in MIDI files set to this parameter.
    velocity_mean=None,
    # Standard deviation(SD) of velocity.
    # This class samples the velocity from a Gaussian distribution of 
    # `velocity_mean` and `velocity_std`.
    # If `None`, the SD of velocity in MIDI files set to this parameter.
    velocity_std=None
)
```

Finally, new MIDI file will be stored in `file_path`.

## References

- Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
- Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
- Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
- Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.

### Related PoC

- [量子力学、統計力学、熱力学における天才物理学者たちの神学的な形象について](https://accel-brain.com/das-theologische-bild-genialer-physiker-in-der-quantenmechanik-und-der-statistischen-mechanik-und-thermodynamik/) (Japanese)
    - [「マクスウェルの悪魔」、力学の基礎法則としての神](https://accel-brain.com/das-theologische-bild-genialer-physiker-in-der-quantenmechanik-und-der-statistischen-mechanik-und-thermodynamik/maxwell-damon/)
- [深層強化学習のベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/) (Japanese)
    - [平均場近似推論の統計力学、自己符号化器としての深層ボルツマンマシン](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/tiefe-boltzmann-maschine-als-selbstkodierer/)
    - [正則化問題における敵対的生成ネットワーク(GANs)と敵対的自己符号化器(AAEs)のネットワーク構造](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/regularisierungsproblem-und-gan/)
- [「人工の理想」を背景とした「万物照応」のデータモデリング](https://accel-brain.com/data-modeling-von-korrespondenz-in-artificial-paradise/) (Japanese)
    - [模倣のアルゴリズムとアルゴリズムの模倣](https://accel-brain.com/data-modeling-von-korrespondenz-in-artificial-paradise/algorithmus-der-mimetischen-innervation/)
    - [叙事的な音楽、マトリックスのリズム](https://accel-brain.com/data-modeling-von-korrespondenz-in-artificial-paradise/epische-musik/)

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
