# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t
from pydbm.synapse_list import Synapse
from pydbm.cnn.layerablecnn.convolutionlayer.deconvolution_layer import DeconvolutionLayer
from pydbm.synapse.cnn_graph import CNNGraph as GivenGraph
from pydbm.synapse.cnn_graph import CNNGraph as InputGraph
from pydbm.synapse.cnn_graph import CNNGraph as ForgotGraph
from pydbm.synapse.cnn_graph import CNNGraph as OutputGraph
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.optimization.opt_params import OptParams
from pydbm.rnn.lstmmodel.conv_lstm_model import ConvLSTMModel


class DeconvLSTMModel(ConvLSTMModel):
    '''
    Deconvolutional LSTM(ConvLSTM).
    
    Convolutional LSTM(ConvLSTM)(Xingjian, S. H. I. et al., 2015), 
    which is a model that structurally couples convolution operators 
    to LSTM networks, can be utilized as components in constructing 
    the Encoder/Decoder. The ConvLSTM is suitable for spatio-temporal data 
    due to its inherent convolutional structure.

    In this class, the convolution and deconvolution are mutually substituted.
    Deconvolution also called transposed convolutions
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_conv_lstm.ipynb
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810)
    '''
    # is-a `Synapse`.
    __graph = None
    
    def get_graph(self):
        ''' getter '''
        if isinstance(self.__graph, Synapse) is False:
            raise TypeError()

        return self.__graph

    def set_graph(self, value):
        ''' setter '''
        if isinstance(value, Synapse) is False:
            raise TypeError()
        self.__graph = value
    
    graph = property(get_graph, set_graph)

    # is-a `OptParams`.
    __opt_params = None

    # Verification function.
    __verificatable_result = None

    # The list of paramters to be differentiated.
    __learned_params_list = []
    
    # Latest loss
    __latest_loss = None
    
    # Latest length of sequneces.
    __cycle_len = 1
    
    # Observed shape of hidden units.
    __hidden_shape = None

    def __init__(
        self,
        graph,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        given_conv=None,
        input_conv=None,
        forgot_conv=None,
        output_conv=None,
        int filter_num=20,
        int channel=3,
        int kernel_size=3,
        double scale=0.1,
        int stride=1,
        int pad=1,
        int seq_len=0,
        int bptt_tau=16,
        double test_size_rate=0.3,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        tol=1e-04,
        tld=100.0,
        pre_learned_dir=None
    ):
        '''
        Init for building LSTM networks.

        Args:
            graph:                          is-a `Synapse`.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            given_conv:                     is-a `ConvolutionLayer` for LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            input_conv:                     is-a `ConvolutionLayer` for input gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            forgot_conv:                    is-a `ConvolutionLayer` for forgot gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            output_conv:                    is-a `ConvolutionLayer` for output gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.
            
            filter_num:                     The number of filter in default settings.
            channel:                        The channel in default settings.
            kernel_size:                    The kernel size in default settings.
            scale:                          Scale of weights and bias vector in convolution layer in default settings.
            stride:                         The stride in default settings.
            pad:                            The pad in default settings.

            seq_len:                        The length of sequences.
                                            If `0`, this model will reference all series elements included 
                                            in observed data points.
                                            If not `0`, only first sequence will be observed by this model 
                                            and will be feedfowarded as feature points.
                                            This parameter enables you to build this class as `Decoder` in
                                            Sequence-to-Sequence(Seq2seq) scheme.

            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
                                            If `0`, this class referes all past data in BPTT.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

            pre_learned_dir:                Path to directory that stores pre-learned parameters.

        '''
        self.graph = graph
        if pre_learned_dir is not None:
            self.graph.load_pre_learned_params(pre_learned_dir + "conv_lstm_graph.npz")

        self.__filter_num = filter_num
        self.__channel = channel
        self.__hidden_shape = None

        if given_conv is None:
            self.__given_conv = DeconvolutionLayer(
                # Computation graph for first convolution layer.
                GivenGraph(
                    # Logistic function as activation function.
                    activation_function=TanhFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__given_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_given_graph.npz"
                )
        else:
            self.__given_conv = given_conv

        if input_conv is None:
            self.__input_conv = DeconvolutionLayer(
                # Computation graph for first convolution layer.
                InputGraph(
                    # Logistic function as activation function.
                    activation_function=LogisticFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__input_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_input_graph.npz"
                )
        else:
            self.__input_conv = input_conv

        if forgot_conv is None:
            self.__forgot_conv = DeconvolutionLayer(
                # Computation graph for first convolution layer.
                ForgotGraph(
                    # Logistic function as activation function.
                    activation_function=LogisticFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__forgot_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_forgot_graph.npz"
                )
        else:
            self.__forgot_conv = forgot_conv

        if output_conv is None:
            self.__output_conv = DeconvolutionLayer(
                # Computation graph for first convolution layer.
                OutputGraph(
                    # Logistic function as activation function.
                    activation_function=TanhFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__output_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_output_graph.npz"
                )
        else:
            self.__output_conv = output_conv

        super().__init__(
            graph=graph,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            given_conv=self.__given_conv,
            input_conv=self.__input_conv,
            forgot_conv=self.__forgot_conv,
            output_conv=self.__output_conv,
            filter_num=filter_num,
            channel=channel,
            kernel_size=kernel_size,
            scale=scale,
            stride=stride,
            pad=pad,
            seq_len=seq_len,
            bptt_tau=bptt_tau,
            test_size_rate=test_size_rate,
            computable_loss=computable_loss,
            opt_params=opt_params,
            verificatable_result=verificatable_result,
            tol=tol,
            tld=tld,
            pre_learned_dir=pre_learned_dir
        )

        logger = getLogger("pydbm")
        self.__logger = logger
