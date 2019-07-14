# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pygan.discriminativemodel.autoencodermodel.convolutional_auto_encoder import ConvolutionalAutoEncoder
from pydbm.cnn.convolutionalneuralnetwork.convolutionalautoencoder.convolutional_ladder_networks import ConvolutionalLadderNetworks as CLN
from pygan.true_sampler import TrueSampler
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConvolutionalLadderNetworks(ConvolutionalAutoEncoder):
    '''
    Ladder Networks with a Stacked convolutional Auto-Encoder as a Discriminator..

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

    def __init__(
        self, 
        convolutional_auto_encoder=None,
        batch_size=10,
        channel=1,
        learning_rate=1e-10,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        opt_params=None,
        feature_matching_layer=0
    ):
        '''
        Init.
        
        Args:
            convolutional_auto_encoder:         is-a `pydbm.cnn.convolutionalneuralnetwork.convolutionalautoencoder.ConvolutionalLadderNetworks`.
            learning_rate:                      Learning rate.
            batch_size:                         Batch size in mini-batch.
            learning_rate:                      Learning rate.
            learning_attenuate_rate:            Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                    Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                                Additionally, in relation to regularization,
                                                this class constrains weight matrixes every `attenuate_epoch`.

            opt_params:                         is-a `pydbm.optimization.opt_params.OptParams`.
            feature_matching_layer:             Key of layer number for feature matching forward/backward.

        '''

        if isinstance(convolutional_auto_encoder, CLN) is False and convolutional_auto_encoder is not None:
            raise TypeError("The type of `convolutional_auto_encoder` must be `pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder.ConvolutionalAutoEncoder`.")

        if opt_params is None:
            opt_params = Adam()
            opt_params.weight_limit = 1e+10
            opt_params.dropout_rate = 0.0

        if convolutional_auto_encoder is None:
            scale = 0.01
            conv1 = ConvolutionLayer1(
                ConvGraph1(
                    activation_function=TanhFunction(),
                    filter_num=batch_size,
                    channel=channel,
                    kernel_size=3,
                    scale=scale,
                    stride=1,
                    pad=1
                )
            )

            conv2 = ConvolutionLayer2(
                ConvGraph2(
                    activation_function=TanhFunction(),
                    filter_num=batch_size,
                    channel=batch_size,
                    kernel_size=3,
                    scale=scale,
                    stride=1,
                    pad=1
                )
            )

            convolutional_auto_encoder = CLN(
                layerable_cnn_list=[
                    conv1, 
                    conv2
                ],
                epochs=100,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=learning_attenuate_rate,
                attenuate_epoch=attenuate_epoch,
                computable_loss=MeanSquaredError(),
                opt_params=opt_params,
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15,
                save_flag=False
            )

        self.__convolutional_auto_encoder = convolutional_auto_encoder
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate

        self.__epoch_counter = 0
        self.__feature_matching_layer = feature_matching_layer
        logger = getLogger("pygan")
        self.__logger = logger

        super().__init__(
            convolutional_auto_encoder=convolutional_auto_encoder,
            batch_size=batch_size,
            channel=channel,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            opt_params=opt_params,
            feature_matching_layer=feature_matching_layer
        )

        self.__alpha_loss_list = []
        self.__sigma_loss_list = []
        self.__mu_loss_list = []

    def inference(self, observed_arr):
        '''
        Draws samples from the `fake` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        result_arr = super().inference(observed_arr)

        alpha_loss = self.__convolutional_auto_encoder.compute_alpha_loss()
        sigma_loss = self.__convolutional_auto_encoder.compute_sigma_loss()
        mu_loss = self.__convolutional_auto_encoder.compute_mu_loss()

        self.__alpha_loss_list.append(alpha_loss)
        self.__sigma_loss_list.append(sigma_loss)
        self.__mu_loss_list.append(mu_loss)

        self.loss = self.loss + alpha_loss + sigma_loss + mu_loss
        return result_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_alpha_loss_arr(self):
        ''' getter '''
        return np.array(self.__alpha_loss_list)

    alpha_loss_arr = property(get_alpha_loss_arr, set_readonly)

    def get_sigma_loss_arr(self):
        ''' getter '''
        return np.array(self.__sigma_loss_list)
    
    sigma_loss_arr = property(get_sigma_loss_arr, set_readonly)

    def get_mu_loss_arr(self):
        ''' getter '''
        return np.array(self.__mu_loss_list)
    
    mu_loss_arr = property(get_mu_loss_arr, set_readonly)
