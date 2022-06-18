# Generative Adversarial Networks Library: pygan

`pygan` is Python library to implement Generative Adversarial Networks(GANs), *Conditional* GANs, Adversarial Auto-Encoders(AAEs), and Energy-based Generative Adversarial Network(EBGAN).

This library makes it possible to design the Generative models based on the Statistical machine learning problems in relation to Generative Adversarial Networks(GANs), *Conditional* GANs, Adversarial Auto-Encoders(AAEs), and Energy-based Generative Adversarial Network(EBGAN) to practice algorithm design for semi-supervised learning. 

## Installation

Install using pip:

```sh
pip install pygan
```

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Generative-Adversarial-Networks](https://github.com/chimera0/accel-brain-code/tree/master/Generative-Adversarial-Networks)

### Python package index(PyPI)

Installers for the latest released version are available at the Python package index.

- [pygan : Python Package Index](https://pypi.python.org/pypi/pygan/)

### Dependencies

- [numpy](https://github.com/numpy/numpy): v1.13.3 or higher.
- [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base): v1.0.0 or higher.
- [mxnet](https://github.com/apache/incubator-mxnet) or [mxnet-cu*](https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/6-use_gpus.html): latest.
  * Only when building a model of this library using [Apache MXNet](https://mxnet.apache.org/).
- [torch](https://pytorch.org/get-started/locally/)
  * Only when building a model of this library using [PyTorch](https://pytorch.org/).

## Documentation

Full documentation is available on [https://code.accel-brain.com/Generative-Adversarial-Networks/](https://code.accel-brain.com/Generative-Adversarial-Networks/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Description

`pygan` is Python library to implement Generative Adversarial Networks(GANs), *Conditional* GANs, Adversarial Auto-Encoders(AAEs), and Energy-based Generative Adversarial Network(EBGAN).

The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes a
min-max adversarial game between two neural networks – a generative model, `G`, and a discriminative
model, `D`. The discriminator model, `D(x)`, is a neural network that computes the probability that
a observed data point `x` in data space is a sample from the data distribution (positive samples) that we are trying to model, rather than a sample from our generative model (negative samples). Concurrently, the generator uses a function `G(z)` that maps samples `z` from the prior `p(z)` to the data space. `G(z)` is trained to maximally confuse the discriminator into believing that samples it generates come from the data distribution. The generator is trained by leveraging the gradient of `D(x)` w.r.t. `x`, and using that to modify its parameters.

### Structural extension for *Conditional* GANs (or cGANs).

The *Conditional* GANs (or cGANs) is a simple extension of the basic GAN model which allows the model to condition on external information. This makes it possible to engage the learned generative model in different "modes" by providing it with different contextual information (Gauthier, J. 2014).

This model can be constructed by simply feeding the data, `y`, to condition on to both the generator and discriminator. In an unconditioned generative model, because the maps samples `z` from the prior `p(z)` are drawn from uniform or normal distribution, there is no control on modes of the data being generated. On the other hand, it is possible to direct the data generation process by conditioning the model on additional information (Mirza, M., & Osindero, S. 2014).

### Structural extension for Adversarial Auto-Encoders(AAEs).

This library also provides the Adversarial Auto-Encoders(AAEs), which is a probabilistic Auto-Encoder that uses GANs to perform variational inference by matching the aggregated posterior of the feature points in hidden layer of the Auto-Encoder with an arbitrary prior distribution(Makhzani, A., et al., 2015). Matching the aggregated posterior to the prior ensures that generating from any part of prior space results in meaningful samples. As a result, the decoder of the Adversarial Auto-Encoder learns a deep generative model that maps the imposed prior to the data distribution.

### Structural extension for Energy-based Generative Adversarial Network(EBGAN).

Reusing the Auto-Encoders, this library introduces the Energy-based Generative Adversarial Network (EBGAN) model(Zhao, J., et al., 2016) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. THe Auto-Encoders have traditionally been used to represent energy-based models. When trained with some regularization terms, the Auto-Encoders have the ability to learn an energy manifold without supervision or negative examples. This means that even when an energy-based Auto-Encoding model is trained to reconstruct a real sample, the model contributes to discovering the data manifold by itself.

### Structural coupling between AAEs and EBGAN.

This library models the Energy-based Adversarial-Auto-Encoder(EBAAE) by structural coupling between AAEs and EBGAN. The learning algorithm equivalents an adversarial training of AAEs as a generator and EBGAN as a discriminator.

### Usecase: Image Generation by GANs.

Import a Python module.

```python
from pygan._mxnet.gan_image_generator import GANImageGenerator
```

Setup logger.

```python
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

logger = getLogger("accelbrainbase")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
```

Initialize `GANImageGenerator`.

```python
gan_image_generator = GANImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
)
```

If you want to use the [PyTorch](https://pytorch.org/) version, import a Python module and initialize `GANImageGenerator`.

```python
from pygan._torch.gan_image_generator import GANImageGenerator
import torch

# Context-manager that changes the selected device.
ctx = "cuda:0" if torch.cuda.is_available() else "cpu"

gan_image_generator = GANImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
    # Context-manager that changes the selected device.
    ctx=ctx,
)
```

Call method `learn`.

```python
gan_image_generator.learn(
    # `int` of the number of training iterations.
    iter_n=100000,
    # `int` of the number of learning of the discriminative model.
    k_step=10,
)
```

You can check logs of posterior.

```python
print(gan_image_generator.GAN.posterior_logs_arr)
```

And, call method `draw`. The generated image data is stored in the variable `arr`.

```python
arr = gan_image_generator.GAN.generative_model.draw()
```

The shape of `arr` is ...
- batch
- channel
- height
- width

For more detailed or original modeling or tuning, see [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base). This library is based on [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).

### Usecase: Image Generation by EBGANs.

Import a Python module.

```python
from pygan._mxnet.ebgan_image_generator import EBGANImageGenerator
```

Setup logger.

```python
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

logger = getLogger("accelbrainbase")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
```

Initialize `EBGANImageGenerator`.

```python
ebgan_image_generator = EBGANImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
)
```
If you want to use the [PyTorch](https://pytorch.org/) version, import a Python module and initialize `EBGANImageGenerator`.

```python
from pygan._torch.ebgan_image_generator import EBGANImageGenerator
import torch

# Context-manager that changes the selected device.
ctx = "cuda:0" if torch.cuda.is_available() else "cpu"

ebgan_image_generator = EBGANImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
    # Context-manager that changes the selected device.
    ctx=ctx,
)
```

Call method `learn`.

```python
ebgan_image_generator.learn(
    # `int` of the number of training iterations.
    iter_n=100000,
    # `int` of the number of learning of the discriminative model.
    k_step=10,
)
```

You can check logs of posterior.

```python
print(ebgan_image_generator.EBGAN.posterior_logs_arr)
```

And, call method `draw`. The generated image data is stored in the variable `arr`.

```python
arr = ebgan_image_generator.EBGAN.generative_model.draw()
```

The shape of `arr` is ...
- batch
- channel
- height
- width

For more detailed or original modeling or tuning, see [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base). This library is based on [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).

### Usecase: Image Generation by AAEs.

Import a Python module.

```python
from pygan._mxnet.ebaae_image_generator import EBAAEImageGenerator
```

Setup a logger.

```python
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

logger = getLogger("accelbrainbase")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
```

Initialize `EBAAEImageGenerator`.

```python
ebaae_image_generator = EBAAEImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
    # `int` of width of image drawn from normal distribution, p(z).
    normal_height=128,
    # `int` of height of image drawn from normal distribution, p(z).
    normal_width=96,
)
```

If you want to use the [PyTorch](https://pytorch.org/) version, import a Python module and initialize `EBAAEImageGenerator`.

```python
from pygan._torch.ebaae_image_generator import EBAAEImageGenerator
import torch

# Context-manager that changes the selected device.
ctx = "cuda:0" if torch.cuda.is_available() else "cpu"

ebaae_image_generator = EBAAEImageGenerator(
    # `list` of path to your directories.
    dir_list=[
        "/path/to/your/image/files/", 
    ],
    # `int` of image width.
    width=128,
    # `int` of image height.
    height=96,
    # `int` of image channel.
    channel=1,
    # `int` of batch size.
    batch_size=40,
    # `float` of learning rate.
    learning_rate=1e-06,
    # `int` of width of image drawn from normal distribution, p(z).
    normal_height=128,
    # `int` of height of image drawn from normal distribution, p(z).
    normal_width=96,
    # Context-manager that changes the selected device.
    ctx=ctx,
)
```

Call method `learn`.

```python
ebaae_image_generator.learn(
    # `int` of the number of training iterations.
    iter_n=100000,
    # `int` of the number of learning of the discriminative model.
    k_step=10,
)
```

You can check logs of posterior.

```python
print(ebaae_image_generator.EBAAE.posterior_logs_arr)
```

And, call method `draw`. The generated image data is stored in the variable `decoded_arr`.

```python
arr_tuple = ebaae_image_generator.EBAAE.generative_model.draw()
feature_points_arr, observed_arr, decoded_arr = arr_tuple
```

The shape of `decoded_arr` is ...
- batch
- channel
- height
- width

For more detailed or original modeling or tuning, see [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base). This library is based on [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).

## References

The basic concepts, theories, and methods behind this library are described in the following books.

<div align="center"><a href="https://www.amazon.co.jp/dp/B08PV4ZQG5/" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Accel-Brain-Books/In-house_R_and_D_in_the_era_of_democratization_of_AI/book_cover.jpg" width="160px" /></a>
  <p>『<a href="https://www.amazon.co.jp/dp/B08PV4ZQG5/ref=sr_1_1?dchild=1&qid=1607343553&s=digital-text&sr=1-1&text=%E6%A0%AA%E5%BC%8F%E4%BC%9A%E7%A4%BEAccel+Brain" target="_blank">「AIの民主化」時代の企業内研究開発: 深層学習の「実学」としての機能分析</a>』(Japanese)</p></div>

<br />
  
<div align="center"><a href="https://www.amazon.co.jp/dp/B093Z533LK" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Accel-Brain-Books/AI_vs_Investors_as_Noise_Traders/book_cover.jpg" width="160px" /></a>
  <p>『<a href="https://www.amazon.co.jp/dp/B093Z533LK" target="_blank">AI vs. ノイズトレーダーとしての投資家たち: 「アルゴリズム戦争」時代の証券投資戦略</a>』(Japanese)</p></div>

<br />

<div align="center"><a href="https://www.amazon.co.jp/dp/B0994CH3CM" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Accel-Brain-Books/Babel_of_Natural_Language_Processing/book_cover.jpg" width="160px" /></a>
  <p>『<a href="https://www.amazon.co.jp/dp/B0994CH3CM" target="_blank">自然言語処理のバベル: 文書自動要約、文章生成AI、チャットボットの意味論</a>』(Japanese)</p></div>

<br />

<div align="center"><a href="https://www.amazon.co.jp/dp/B09C4KYZBX" target="_blank"><img src="https://storage.googleapis.com/accel-brain-code/Accel-Brain-Books/Origin_of_the_statistical_machine_learning/book_cover.jpg" width="160px" /></a>
  <p>『<a href="https://www.amazon.co.jp/dp/B09C4KYZBX" target="_blank">統計的機械学習の根源: 熱力学、量子力学、統計力学における天才物理学者たちの神学的な理念</a>』(Japanese)</p></div>


Specific references are the following papers and books.

- Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
- Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
- Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
- Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
- Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
- Mogren, O. (2016). C-RNN-GAN: Continuous recurrent neural networks with adversarial training. arXiv preprint arXiv:1611.09904.
- Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011, June). Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of the 28th International Conference on International Conference on Machine Learning (pp. 833-840). Omnipress.
- Rifai, S., Mesnil, G., Vincent, P., Muller, X., Bengio, Y., Dauphin, Y., & Glorot, X. (2011, September). Higher order contractive auto-encoder. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645-660). Springer, Berlin, Heidelberg.
- Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).
- Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.
- Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
- Warde-Farley, D., & Bengio, Y. (2016). Improving generative adversarial networks with denoising feature matching.

## Author

- accel-brain

## Author URI

- https://accel-brain.co.jp/
- https://accel-brain.com/

## License

- GNU General Public License v2.0
