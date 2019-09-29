# -*- coding: utf-8 -*-
import numpy as np
from pygan.feature_matching import FeatureMatching
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder


class DenoisingFeatureMatching(FeatureMatching):
    '''
    Value function with Feature matching, which addresses the instability of GANs 
    by specifying a new objective for the generator that prevents it from overtraining 
    on the current discriminator(Salimans, T., et al., 2016).
    
    "Instead of directly maximizing the output of the discriminator, 
    the new objective requires the generator to generate data that matches
    the statistics of the real data, where we use the discriminator only to specify 
    the statistics that we think are worth matching." (Salimans, T., et al., 2016, p2.)

    While the gradient of the loss function defined by the discriminator may be a source of information
    mostly relevant to very local improvements, the discriminator itself is a potentially valuable source
    of compact descriptors of the training data. Although non-stationary, the distribution of the highlevel 
    activations of the discriminator when evaluated on data is ripe for exploitation as an additional source 
    of knowledge about salient aspects of the data distribution. Warde-Farley, D., et al. (2016) proposed in 
    this work to track this distribution with a denoising auto-encoder trained on the discriminator’s hidden 
    states when evaluated on training data.

    Then Warde-Farley, D., et al. (2016) proposed an augmented training procedure for generative adversarial networks
    designed to address shortcomings of the original by directing the generator towards probable configurations of abstract 
    discriminator features. They estimate and track the distribution of these features, as computed from data, with a 
    denoising auto-encoder, and use it to propose high-level targets for the generator.

    References:
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.
        - Warde-Farley, D., & Bengio, Y. (2016). Improving generative adversarial networks with denoising feature matching.
    '''

    def __init__(
        self, 
        auto_encoder,
        lambda1=0.5, 
        lambda2=0.0, 
        lambda3=0.5,
        computable_loss=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        noising_f=None
    ):
        '''
        Init.

        Args:
            auto_encoder:               A denoising Auto-Encoder whose type is ...
                                        - `pydbm.nn.simple_auto_encoder.SimpleAutoEncoder`

            lambda1:                    Trade-off parameter. This means a weight for results of standard feature matching.
            lambda2:                    Trade-off parameter. This means a weight for results of difference between generated data points and true samples.
            lambda3:                    Trade-off parameter. This means a weight for results of delta of denoising Auto-Encoder.
            computable_loss:            is-a `pydbm.loss.interface.computable_loss.ComputableLoss`.
                                        If `None`, the default value is a `MeanSquaredError`.

            learning_rate:              Learning rate.
            learning_attenuate_rate:    Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:            Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                        Additionally, in relation to regularization,
                                        this class constrains weight matrixes every `attenuate_epoch`.

            noising_f:                  Noise function that receives variable generated `np.ndarray`. 
                                        If `None`, this value equivalents of `np.random.normal(loc=0, scale=1.0, size=(hoge, fuga))`.

        Exceptions:
            ValueError:     When the sum of `lambda1` and `lambda2` is not less than `1.0`.
                            Those parameters are trade-off parameters.
        '''
        if isinstance(auto_encoder, SimpleAutoEncoder) is False:
            raise TypeError("The type of `pydbm.nn.simple_auto_encoder.SimpleAutoEncoder`.")

        if lambda3 <= 0.0:
            raise ValueError("The value of `lambda3` must be more than `0.0`.")

        if lambda1 + lambda2 + lambda3 > 1:
            raise ValueError("The sum of `lambda1` and `lambda2` and `lambda3` must be less than `1.0`. Those parameters are trade-off parameters.")

        self.__auto_encoder = auto_encoder
        self.__lambda3 = lambda3
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__epoch_counter = 0

        if noising_f is None:
            def noising_f(arr):
                return arr + np.random.normal(size=arr.shape)

        self.__noising_f = noising_f

        super().__init__(
            lambda1=lambda1, 
            lambda2=lambda2, 
            computable_loss=computable_loss,
        )

    def compute_delta(
        self,
        true_sampler, 
        discriminative_model,
        generated_arr
    ):
        '''
        Compute generator's reward.

        Args:
            true_sampler:           Sampler which draws samples from the `true` distribution.
            discriminative_model:   Discriminator which discriminates `true` from `fake`.
            generated_arr:          `np.ndarray` generated data points.
        
        Returns:
            `np.ndarray` of Gradients.
        '''
        grad_arr = super().compute_delta(
            true_sampler=true_sampler, 
            discriminative_model=discriminative_model,
            generated_arr=generated_arr
        )

        if ((self.__epoch_counter + 1) % self.__attenuate_epoch == 0):
            self.__learning_rate = self.__learning_rate * self.__learning_attenuate_rate

        _generated_arr = discriminative_model.feature_matching_forward(generated_arr)
        shape_tuple = _generated_arr.shape
        _noised_arr = self.__noising_f(_generated_arr)
        _noised_arr = _noised_arr.reshape((
            _noised_arr.shape[0],
            -1
        ))

        _reconstructed_arr = self.__auto_encoder.inference(_noised_arr)
        _reconstructed_arr = _reconstructed_arr.reshape(shape_tuple)

        grad_arr3 = self.__auto_encoder.computable_loss.compute_delta(
            _reconstructed_arr,
            _generated_arr
        )
        grad_arr3 = grad_arr3.reshape((
            grad_arr3.shape[0],
            -1
        ))
        grad_arr3 = self.__auto_encoder.back_propagation(grad_arr3)
        self.__auto_encoder.optimize(
            learning_rate=self.__learning_rate,
            epoch=self.__epoch_counter
        )
        self.__epoch_counter += 1

        grad_arr3 = grad_arr3.reshape(shape_tuple)
        grad_arr3 = discriminative_model.feature_matching_backward(grad_arr3)
        grad_arr3 = grad_arr3.reshape(generated_arr.shape)

        loss3 = self.computable_loss.compute_loss(
            _reconstructed_arr,
            _generated_arr
        )
        grad_arr = grad_arr + (grad_arr3 * self.__lambda3)
        self._loss_list[-1] = self._loss_list[-1] + (loss3 * self.__lambda3)

        return grad_arr
