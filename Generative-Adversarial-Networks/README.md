# Generative Adversarial Networks Library: pygan

`pygan` is Python library to implement Generative Adversarial Networks(GANs) and Adversarial Auto-Encoders(AAEs).

This library makes it possible to design the Generative models based on the Statistical machine learning problems in relation to Generative Adversarial Networks(GANs) and Adversarial Auto-Encoders(AAEs) to practice algorithm design for semi-supervised learning. But this library provides components for designers, not for end-users of state-of-the-art black boxes. Briefly speaking the philosophy of this library, *give user hype-driven blackboxes and you feed him for a day; show him how to design algorithms and you feed him for a lifetime.* So algorithm is power.

Now this library is **alpha version**.

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

- [pyqlearning : Python Package Index](https://pypi.python.org/pypi/pygan/)

### Dependencies

- numpy: v1.13.3 or higher.

#### Option

- [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern): v1.4.0 or higher.
    * Only if you want to implement the components based on this library.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Generative-Adversarial-Networks/](https://code.accel-brain.com/Generative-Adversarial-Networks/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Description

`pygan` is Python library to implement Generative Adversarial Networks(GANs) and Adversarial Auto-Encoders(AAEs).

The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes a
min-max adversarial game between two neural networks – a generative model, `G`, and a discriminative
model, `D`. The discriminator model, `D(x)`, is a neural network that computes the probability that
a observed data point `x` in data space is a sample from the data distribution (positive samples) that we are trying to model, rather than a sample from our generative model (negative samples). Concurrently, the generator uses a function `G(z)` that maps samples `z` from the prior `p(z)` to the data space. `G(z)` is trained to maximally confuse the discriminator into believing that samples it generates come from the data distribution. The generator is trained by leveraging the gradient of `D(x)` w.r.t. `x`, and using that to modify its parameters.

This library provides the Adversarial Auto-Encoders(AAEs), which is a probabilistic Auto-Encoder that uses GANs to perform variational inference by matching the aggregated posterior of the feature points in hidden layer of the Auto-Encoder with an arbitrary prior distribution(Makhzani, A., et al., 2015). Matching the aggregated posterior to the prior ensures that generating from any part of prior space results in meaningful samples. As a result, the decoder of the Adversarial Auto-Encoder learns a deep generative model that maps the imposed prior to the data distribution.


## References

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
- Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
