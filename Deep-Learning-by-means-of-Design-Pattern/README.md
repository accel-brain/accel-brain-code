# Deep Learning Library: pydbm

`pydbm` is Python3 library for building restricted boltzmann machine, deep boltzmann machine, and multi-layer neural networks.

This is Cython version. [pydbm_mxnet](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern/mxnet) (MXNet version) is derived from this library.

## Description

The function of this library is building and modeling restricted boltzmann machine, deep boltzmann machine, and multi-layer neural networks. The models are functionally equivalent to stacked auto-encoder. The main function is the same as dimensions reduction(or pre-training).

## Documentation

Full documentation is available on [https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/](https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Installation

Install using pip:

```sh
pip install pydbm
```

Or, you can install from wheel file.

```sh
pip install https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/pydbm-1.1.5-cp36-cp36m-linux_x86_64.whl
```

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Deep-Learning-by-means-of-Design-Pattern](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern)

### Python package index(PyPI)

Installers for the latest released version are available at the Python package index.

- [pydbm : Python Package Index](https://pypi.python.org/pypi/pydbm)

### Dependencies

- numpy: v1.13.3 or higher.
- cython: v0.27.1 or higher.

## Usecase: Building the deep boltzmann machine for feature extracting.

Import Python and Cython modules.

```python
# The `Client` in Builder Pattern
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
# The `Concrete Builder` in Builder Pattern.
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
# Contrastive Divergence for function approximation.
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# ReLu Function as activation function.
from pydbm.activation.relu_function import ReLuFunction

```

Instantiate objects and call the method.

```python
dbm = DeepBoltzmannMachine(
    DBMMultiLayerBuilder(),
    # Dimention in visible layer, hidden layer, and second hidden layer.
    [traning_x.shape[1], 10, traning_x.shape[1]],
    [ReLuFunction(), LogisticFunction(), TanhFunction()], # Setting objects for activation function.
    [ContrastiveDivergence(), ContrastiveDivergence()],   # Setting the object for function approximation.
    0.05, # Setting learning rate.
    0.5   # Setting dropout rate.
)
# Execute learning.
dbm.learn(traning_arr, traning_count=1000)
```

And the feature points can be extracted by this method.

```python
print(dbm.get_feature_point_list(0))
```

## Usecase: Extracting all feature points for dimensions reduction(or pre-training)

Import Python and Cython modules.

```python
# `StackedAutoEncoder` is-a `DeepBoltzmannMachine`.
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
# The `Concrete Builder` in Builder Pattern.
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
# Contrastive Divergence for function approximation.
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
```

Instantiate objects and call the method.

```python
# Setting objects for activation function.
activation_list = [LogisticFunction(), LogisticFunction(), LogisticFunction()]
# Setting the object for function approximation.
approximaion_list = [ContrastiveDivergence(), ContrastiveDivergence()]

dbm = StackedAutoEncoder(
    DBMMultiLayerBuilder(),
    [target_arr.shape[1], 10, target_arr.shape[1]],
    activation_list,
    approximaion_list,
    0.05, # Setting learning rate.
    0.5   # Setting dropout rate.
)

# Execute learning.
dbm.learn(target_arr, traning_count=1)
```

And the result of dimention reduction can be extracted by this property.

```python
pre_trained_arr = dbm.feature_points_arr
```

### Extract pre-training weights

If you want to get the pre-training weights, call `get_weight_arr_list` method.

```python
weight_arr_list = dbm.get_weight_arr_list()
```
`weight_arr_list` is the `list` of weights of each links in DBM. `weight_arr_list[0]` is 2-d `np.ndarray` of weights between visible layer and first hidden layer.

### Extract reconstruction error rate.

You can check the reconstruction error rate. During the approximation of the Contrastive Divergence, the mean squared error(MSE) between the observed data points and the activities in visible layer is computed as the reconstruction error rate.

Call `get_reconstruct_error_arr` method as follow.

```python
reconstruct_error_arr = dbm.get_reconstruct_error_arr(layer_number=0)
```

`layer_number` corresponds to the index of `approximaion_list`. And `reconstruct_error_arr` is the `np.ndarray` of reconstruction error rates.

### Performance

Run a program: [demo_stacked_auto_encoder.py](https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo_stacked_auto_encoder.py)

```sh
time python demo_stacked_auto_encoder.py
```

The result is follow.
 
```sh
real    1m35.472s
user    1m32.300s
sys     0m3.136s
```

#### Detail

This experiment was performed under the following conditions.

##### Machine type

- vCPU: `2`
- memory: `8GB`
- CPU Platform: Intel Ivy Bridge

##### Observation Data Points

The observated data is the result of `np.random.uniform(size=(10000, 10000))`.

##### Number of units

- Visible layer: `10000`
- hidden layer(feature point): `10`
- hidden layer: `10000`

##### Activation functions

- visible:                Logistic Function
- hidden(feature point):  Logistic Function
- hidden:                 Logistic Function

##### Approximation

- Contrastive Divergence

##### Hyper parameters

- Learning rate: `0.05`
- Dropout rate: `0.5`

##### Feature points

```
0.190599  0.183594  0.482996  0.911710  0.939766  0.202852  0.042163
0.470003  0.104970  0.602966  0.927917  0.134440  0.600353  0.264248
0.419805  0.158642  0.328253  0.163071  0.017190  0.982587  0.779166
0.656428  0.947666  0.409032  0.959559  0.397501  0.353150  0.614216
0.167008  0.424654  0.204616  0.573720  0.147871  0.722278  0.068951
.....
```

##### Reconstruct error

```
 [ 0.08297197  0.07091231  0.0823424  ...,  0.0721624   0.08404181  0.06981017]
```

### Design thought

In relation to my [Automatic Summarization Library](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization), it is important for me that the models are functionally equivalent to stacked auto-encoder. The main function I observe is the same as dimensions reduction(or pre-training). But the functional reusability of the models can be not limited to this. These Python Scripts can be considered a kind of *experiment result* to verify effectiveness of object-oriented analysis, object-oriented design, and GoF's design pattern in designing and modeling neural network, deep learning, and [reinforcement-Learning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning).

For instance, [dbm_multi_layer_builder.pyx](https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/pydbm/dbm/builders/dbm_multi_layer_builder.pyx) is implemented for running the **deep boltzmann machine** to extract so-called feature points. This script is premised on a kind of *builder pattern* for separating the construction of complex **restricted boltzmann machines** from its **graph** representation so that the same construction process can create different representations. Because of common design pattern and polymorphism, the **stacked auto-encoder** in [demo_stacked_auto_encoder.py](https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo_stacked_auto_encoder.py) is *functionally equivalent* to **deep boltzmann machine**.

### More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/) (Japanese)
    - Implemented by the `C++` version of this library, these 20001 bots are able to execute the dimensions reduction(or pre-training) for natural language processing to run as 20001 web-crawlers and 20001 web-scrapers.
- [ハッカー倫理に準拠した人工知能のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/) (Japanese)
    - [プロトタイプの開発：深層強化学習のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/5/#i-2)

### Related PoC

- [Webクローラ型人工知能によるパラドックス探索暴露機能の社会進化論](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/) (Japanese)
    - [プロトタイプの開発：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8)
- [深層強化学習のベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/) (Japanese)
    - [プロトタイプの開発：深層学習と強化学習による「排除された第三項」の推論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/4/#i-5)
- [ハッカー倫理に準拠した人工知能のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/) (Japanese)
    - [プロトタイプの開発：深層強化学習のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/5/#i-2)
- [ヴァーチャルリアリティにおける動物的「身体」の物神崇拝的なユースケース](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/) (Japanese)
    - [プロトタイプの開発：「人工天使ヒューズ＝ヒストリア」](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/4/#i-6)

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0

## References

- Ackley, D. H., Hinton, G. E., &amp; Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
- Le Roux, N., &amp; Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
- Salakhutdinov, R., &amp; Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).
