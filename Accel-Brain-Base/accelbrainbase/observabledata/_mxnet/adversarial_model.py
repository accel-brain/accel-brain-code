# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from abc import abstractproperty


class AdversarialModel(ObservableData):
    '''
    The abstract class to build the Generative Adversarial Networks(GANs).

    The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework 
    establishes a min-max adversarial game between two neural networks – a generative 
    model, `G`, and a discriminative model, `D`. The discriminator model, 
    `D(x)`, is a neural network that computes the probability that a observed 
    data point `x` in data space is a sample from the data distribution (positive samples) 
    that we are trying to model, rather than a sample from our generative model 
    (negative samples). Concurrently, the generator uses a function `G(z)` that maps 
    samples `z` from the prior `p(z)` to the data space. `G(z)` is trained to maximally 
    confuse the discriminator into believing that samples it generates come from the 
    data distribution. The generator is trained by leveraging the gradient of `D(x)` w.r.t. 
    `x`, and using that to modify its parameters.

    References:
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
    '''

    @abstractproperty
    def model(self):
        ''' is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        raise NotImplementedError()

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input images.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        self.model.learn(iteratable_data)

    def inference(self, observed_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self.model.inference(observed_arr)

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`. '''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
