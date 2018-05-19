# Deep Learning Library: pydbm

`pydbm` is Python library for building Restricted Boltzmann Machine(RBM), Deep Boltzmann Machine(DBM), Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent neural network Restricted Boltzmann Machine(RNN-RBM), and Shape Boltzmann Machine(Shape-BM).

This is Cython version. [pydbm_mxnet](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern/mxnet) (MXNet version) is derived from this library.

## Description

`pydbm` is Python library for building Restricted Boltzmann Machine(RBM), Deep Boltzmann Machine(DBM), Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent neural network Restricted Boltzmann Machine(RNN-RBM), and Shape Boltzmann Machine(Shape-BM). This is **Cython version**. [pydbm_mxnet](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern/mxnet) (MXNet version) is derived from this library.

The function of this library is building and modeling Restricted Boltzmann Machine(RBM) and Deep Boltzmann Machine(DBM). The models are functionally equivalent to stacked auto-encoder. The basic function is the same as dimensions reduction(or pre-training). And this library enables you to build many functional extensions from RBM and DBM such as Recurrent Temporal Restricted Boltzmann Machine(RTRBM), Recurrent Neural Network Restricted Boltzmann Machine(RNN-RBM), and Shape Boltzmann Machine(Shape-BM).

As more usecases, RTRBM and RNN-RBM can learn dependency structures in temporal patterns such as music, natural sentences, and n-gram. RTRBM is a probabilistic time-series model which can be viewed as a temporal stack of RBMs, where each RBM has a contextual hidden state that is received from the previous RBM and is used to modulate its hidden units bias. The RTRBM can be understood as a sequence of conditional RBMs whose parameters are the output of a deterministic RNN, with the constraint that the hidden units must describe the conditional distributions. This constraint can be lifted by combining a full RNN with distinct hidden units. In terms of this possibility, RNN-RBM is structurally expanded model from RTRBM that allows more freedom to describe the temporal dependencies involved.

On the other hand, the usecases of Shape-BM are image segmentation, object detection, inpainting and graphics. Shape-BM is the model for the task of modeling binary shape images, in that samples from the model look realistic and it can generalize to generate samples that differ from training examples.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/](https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Installation

Install using pip:

```sh
pip install pydbm
```

Or, you can install from wheel file.

```sh
pip install https://storage.googleapis.com/accel-brain-code/Deep-Learning-by-means-of-Design-Pattern/pydbm-{X.X.X}-cp36-cp36m-linux_x86_64.whl
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
dbm.learn(
    traning_arr
    1, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=200,  # Batch size in mini-batch training.
    r_batch_size=-1  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
)
```

If you do not want to execute the mini-batch training, the value of `batch_size` must be `-1`. And `r_batch_size` is also parameter to control the mini-batch training but is refered only in inference and reconstruction. If this value is more than `0`,  the inferencing is a kind of reccursive learning with the mini-batch training.

And the feature points can be extracted by this method.

```python
print(dbm.get_feature_point_list(0))
```

## Usecase: Image segmentation by Shape-BM.

First, acquire image data and binarize it.

```python
from PIL import Image
img = Image.open("horse099.jpg")
img
```

```python
img_bin = img.convert("1")
img_bin
```

Set up hyperparameters.

```python
overlap_n = 4
learning_rate = 0.01
```

`overlap_n` is hyperparameter specific to Shape-BM. In the visible layer, this model has so-called local receptive fields by connecting each first hidden unit only to a subset of the visible units, corresponding to one of four square patches. Each patch overlaps its neighbor by `overlap_n` pixels (Eslami, S. A., et al, 2014).

And import Python and Cython modules.

```python
# The `Client` in Builder Pattern
from pydbm.dbm.deepboltzmannmachine.shape_boltzmann_machine import ShapeBoltzmannMachine
# The `Concrete Builder` in Builder Pattern.
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
```

Instantiate objects and call the method.

```python
dbm = ShapeBoltzmannMachine(
    DBMMultiLayerBuilder(),
    learning_rate=learning_rate,
    overlap_n=overlap_n
)

img_arr = np.asarray(img_bin)
img_arr = img_arr.astype(np.float64)

# Execute learning.
dbm.learn(
    img_arr, # `np.ndarray` of image data.
    1, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=300,  # Batch size in mini-batch training.
    r_batch_size=-1,  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
    sgd_flag=True
)
```

Extract `dbm.visible_points_arr` as the observed data points in visible layer. This `np.ndarray` is segmented image data.

```python
inferenced_data_arr = dbm.visible_points_arr.copy()
inferenced_data_arr = 255 - inferenced_data_arr
Image.fromarray(np.uint8(inferenced_data_arr))
```

## Usecase: Building the Recurrent Temporal Restricted Boltzmann Machine for recursive learning.

Import Python and Cython modules.

```python
# `Builder` in `Builder Patter`.
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
# The object of Restricted Boltzmann Machine.
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
# RNN and Contrastive Divergence for function approximation.
from pydbm.approximation.rt_rbm_cd import RTRBMCD
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Softmax Function as activation function.
from pydbm.activation.softmax_function import SoftmaxFunction
```

Instantiate objects and execute learning.

```python
# `Builder` in `Builder Pattern` for RTRBM.
rtrbm_builder = RTRBMSimpleBuilder()
# Learning rate.
rtrbm_builder.learning_rate = 0.00001
# Set units in visible layer.
rtrbm_builder.visible_neuron_part(LogisticFunction(), arr.shape[1])
# Set units in hidden layer.
rtrbm_builder.hidden_neuron_part(LogisticFunction(), 3)
# Set units in RNN layer.
rtrbm_builder.rnn_neuron_part(LogisticFunction())
# Set graph and approximation function.
rtrbm_builder.graph_part(RTRBMCD())
# Building.
rbm = rtrbm_builder.get_result()

# Learning.
for i in range(arr.shape[0]):
    rbm.approximate_learning(
        arr[i],
        traning_count=1, 
        batch_size=200
    )
```

The `rbm` has a `np.ndarray` of `graph.visible_activity_arr`. The `graph.visible_activity_arr` is the inferenced feature points. This value can be observed as data point.

```python
test_arr = arr[0]
result_list = [None] * arr.shape[0]
for i in range(arr.shape[0]):
    # Execute recursive learning.
    rbm.approximate_inferencing(
        test_arr,
        traning_count=1, 
        r_batch_size=-1
    )
    # The feature points can be observed data points.
    result_list[i] = test_arr = rbm.graph.visible_activity_arr

print(np.array(result_list))
```

## Usecase: Building the Recurrent Neural Network Restricted Boltzmann Machine for recursive learning.

Import Python and Cython modules.

```python
# `Builder` in `Builder Patter`.
from pydbm.dbm.builders.rnn_rbm_simple_builder import RNNRBMSimpleBuilder
# The object of Restricted Boltzmann Machine.
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
# RNN and Contrastive Divergence for function approximation.
from pydbm.approximation.rtrbmcd.rnn_rbm_cd import RNNRBMCD
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Softmax Function as activation function.
from pydbm.activation.softmax_function import SoftmaxFunction
```

Instantiate objects and execute learning.

```python
# `Builder` in `Builder Pattern` for RNN-RBM.
rnnrbm_builder = RNNRBMSimpleBuilder()
# Learning rate.
rnnrbm_builder.learning_rate = 0.00001
# Set units in visible layer.
rnnrbm_builder.visible_neuron_part(LogisticFunction(), arr.shape[1])
# Set units in hidden layer.
rnnrbm_builder.hidden_neuron_part(LogisticFunction(), 3)
# Set units in RNN layer.
rnnrbm_builder.rnn_neuron_part(LogisticFunction())
# Set graph and approximation function.
rnnrbm_builder.graph_part(RNNRBMCD())
# Building.
rbm = rnnrbm_builder.get_result()

# Learning.
for i in range(arr.shape[0]):
    rbm.approximate_learning(
        arr[i],
        traning_count=1, 
        batch_size=200
    )
```

The `rbm` has a `np.ndarray` of `graph.visible_activity_arr`. The `graph.visible_activity_arr` is the inferenced feature points. This value can be observed as data point.

```python
test_arr = arr[0]
result_list = [None] * arr.shape[0]
for i in range(arr.shape[0]):
    # Execute recursive learning.
    rbm.approximate_inferencing(
        test_arr,
        traning_count=1, 
        r_batch_size=-1
    )
    # The feature points can be observed data points.
    result_list[i] = test_arr = rbm.graph.visible_activity_arr

print(np.array(result_list))
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
dbm.learn(
    target_arr,
    1, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=200,  # Batch size in mini-batch training.
    r_batch_size=-1  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
)
```

### Extract the result of dimention reduction

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

## References

- Ackley, D. H., Hinton, G. E., &amp; Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
- Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
- Eslami, S. A., Heess, N., Williams, C. K., & Winn, J. (2014). The shape boltzmann machine: a strong model of object shape. International Journal of Computer Vision, 107(2), 155-176.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
- Le Roux, N., &amp; Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
- Salakhutdinov, R., &amp; Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).
- Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).

## Design thought

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
