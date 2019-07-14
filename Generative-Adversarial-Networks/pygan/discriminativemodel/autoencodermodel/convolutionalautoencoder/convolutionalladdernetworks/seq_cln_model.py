# -*- coding: utf-8 -*-
import numpy as np
from pygan.discriminativemodel.autoencodermodel.convolutionalautoencoder.convolutional_ladder_networks import ConvolutionalLadderNetworks


class SeqCLNModel(ConvolutionalLadderNetworks):
    '''
    Ladder Networks with a Stacked convolutional Auto-Encoder as a Discriminator..

    This model observes sequencal data as image-like data.

    If the length of sequence is `T` and the dimension is `D`, 
    image-like matrix will be configured as a `T` × `D` matrix.

    References:
        - Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. In Advances in neural information processing systems (pp. 153-160).
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Erhan, D., Bengio, Y., Courville, A., Manzagol, P. A., Vincent, P., & Bengio, S. (2010). Why does unsupervised pre-training help deep learning?. Journal of Machine Learning Research, 11(Feb), 625-660.
        - Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures. Department dInformatique et Recherche Operationnelle, University of Montreal, QC, Canada, Tech. Rep, 1355, 1.
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning (adaptive computation and machine learning series). Adaptive Computation and Machine Learning series, 800.
        - Manisha, P., & Gujar, S. (2018). Generative Adversarial Networks (GANs): What it can generate and What it cannot?. arXiv preprint arXiv:1804.00140.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. In Advances in neural information processing systems (pp. 3546-3554).
        - Valpola, H. (2015). From neural PCA to deep unsupervised learning. In Advances in Independent Component Analysis and Learning Machines (pp. 143-171). Academic Press.
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    # Add channel or not.
    __add_channel_flag = False

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        if observed_arr.ndim < 4:
            # Add rank for channel.
            observed_arr = np.expand_dims(observed_arr, axis=1)
            self.__add_channel_flag = True
        else:
            self.__add_channel_flag = False

        return super().inference(observed_arr)

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        delta_arr = super().learn(grad_arr, fix_opt_flag)
        if self.__add_channel_flag is True:
            return delta_arr[:, 0]
        else:
            return delta_arr

    def feature_matching_forward(self, observed_arr):
        '''
        Forward propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        if observed_arr.ndim < 4:
            # Add rank for channel.
            observed_arr = np.expand_dims(observed_arr, axis=1)

        return super().feature_matching_forward(observed_arr)
