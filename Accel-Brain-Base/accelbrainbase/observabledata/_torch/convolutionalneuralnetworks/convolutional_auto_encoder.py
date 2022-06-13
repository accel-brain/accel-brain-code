# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.observabledata._torch.convolutional_neural_networks import ConvolutionalNeuralNetworks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam

import numpy as np
from logging import getLogger


class ConvolutionalAutoEncoder(ConvolutionalNeuralNetworks):
    '''
    Convolutional Auto-Encoder.

    A stack of Convolutional Auto-Encoder (Masci, J., et al., 2011) 
    forms a convolutional neural network(CNN), which are among the most successful models 
    for supervised image classification.  Each Convolutional Auto-Encoder is trained 
    using conventional on-line gradient descent without additional regularization terms.
    
    In this library, Convolutional Auto-Encoder is also based on Encoder/Decoder scheme.
    The encoder is to the decoder what the Convolution is to the Deconvolution.
    The Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        initializer_f=None,
        optimizer_f=None,
        encoder_optimizer_f=None,
        decoder_optimizer_f=None,
        learning_rate=1e-05,
        hidden_units_list=[],
        output_nn=None,
        hidden_dropout_rate_list=[],
        hidden_activation_list=[],
        hidden_batch_norm_list=[],
        ctx="cpu",
        regularizatable_data_list=[],
        scale=1.0,
        tied_weights_flag=True,
        init_deferred_flag=None,
        not_init_flag=False,
        wd=None,
    ):
        '''
        Init.

        Args:
            encoder:                        is-a `CNNHybrid`.
            decoder:                        is-a `CNNHybrid`.
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer_f:                  is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.

            hidden_units_list:              `list` of `mxnet.gluon.nn._conv` in hidden layers.
            output_nn:                      is-a `NNHybrid` as output layers.
                                            If `None`, last layer in `hidden_units_list` will be considered as an output layer.

            hidden_dropout_rate_list:       `list` of `float` of dropout rate in hidden layers.

            optimizer_name:                 `str` of name of optimizer.

            hidden_activation_list:         `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm` in hidden layers.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            regularizatable_data_list:      `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            tied_weights_flag:              `bool` of flag to tied weights or not.
            wd:                             `float` of parameter of weight decay.
            init_deferred_flag:             `bool` that means initialization in this class will be deferred or not.
        '''
        if isinstance(encoder, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `encoder` must be `ConvolutionalNeuralNetworks`.")
        if isinstance(decoder, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `decoder` must be `ConvolutionalNeuralNetworks`.")

        if len(hidden_units_list) != len(hidden_activation_list):
            raise ValueError("The length of `hidden_units_list` and `hidden_activation_list` must be equivalent.")

        if len(hidden_dropout_rate_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_dropout_rate_list` and `hidden_units_list` must be equivalent.")

        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        if init_deferred_flag is None:
            init_deferred_flag = self.init_deferred_flag
        elif isinstance(init_deferred_flag, bool) is False:
            raise TypeError("The type of `init_deferred_flag` must be `bool`.")

        self.__not_init_flag = not_init_flag
        self.init_deferred_flag = True

        super().__init__(
            computable_loss=computable_loss,
            initializer_f=initializer_f,
            optimizer_f=optimizer_f,
            learning_rate=learning_rate,
            hidden_units_list=hidden_units_list,
            output_nn=output_nn,
            hidden_dropout_rate_list=hidden_dropout_rate_list,
            hidden_activation_list=hidden_activation_list,
            hidden_batch_norm_list=hidden_batch_norm_list,
            ctx=ctx,
            regularizatable_data_list=regularizatable_data_list,
            scale=scale,
        )
        self.init_deferred_flag = init_deferred_flag
        self.encoder = encoder
        self.decoder = decoder
        self.__tied_weights_flag = tied_weights_flag
        self.output_nn = output_nn
        self.optimizer_f = optimizer_f
        self.encoder_optimizer_f = encoder_optimizer_f
        self.decoder_optimizer_f = decoder_optimizer_f
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.output_optimizer = None
        self.__ctx = ctx

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if self.encoder_optimizer_f is not None:
                    self.encoder_optimizer = self.encoder_optimizer_f(
                        self.encoder.parameters(), 
                    )
                elif self.optimizer_f is not None:
                    self.encoder_optimizer = self.optimizer_f(
                        self.encoder.parameters(), 
                    )
                else:
                    self.encoder_optimizer = Adam(
                        self.encoder.parameters(), 
                        lr=self.__learning_rate,
                    )

                if self.decoder_optimizer_f is not None:
                    self.decoder_optimizer = self.decoder_optimizer_f(
                        self.decoder.parameters(), 
                    )
                elif self.optimizer_f is not None:
                    self.decoder_optimizer = self.optimizer_f(
                        self.decoder.parameters(), 
                    )
                else:
                    self.decoder_optimizer = Adam(
                        self.decoder.parameters(), 
                        lr=self.__learning_rate,
                    )

    def parameters(self):
        '''
        '''
        params_dict_list = [
            {
                "params": self.encoder.parameters(),
            },
            {
                "params": self.decoder.parameters(),
            }
        ]
        if self.output_nn is not None:
            params_dict_list.append(
                {
                    "params": self.output_nn.parameters()
                }
            )
        return params_dict_list

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input images.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        learning_rate = self.__learning_rate
        try:
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                self.batch_size = batch_observed_arr.shape[0]
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # rank-3
                pred_arr = self.inference(batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        # rank-3
                        test_pred_arr = self.inference(test_batch_observed_arr)
                        test_loss = self.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr
                        )
                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()

                    self.__loss_list.append((_loss, _test_loss))

                    self.__logger.debug(
                        "Epochs: " + str(epoch + 1) + " Train loss: " + str(_loss) + " Test loss: " + str(_test_loss)
                    )
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def inference(self, observed_arr):
        '''
        Inference the feature points to reconstruct the observed data points.

        Args:
            observed_arr:           rank-4 array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, channel, height, width)

        Returns:
            `tensor` of inferenced feature points.
        '''
        return self(observed_arr)

    def compute_loss(self, pred_arr, labeled_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       `tensor`.
            labeled_arr:    `tensor`.

        Returns:
            loss.
        '''
        return self.__computable_loss(pred_arr, labeled_arr)

    def extract_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it.

        Returns:
            The `tensor` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return self.feature_points_arr

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_arr_dict = {}

        params_dict = self.encoder.extract_learned_dict()
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        params_dict = self.decoder.extract_learned_dict()
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

    def forward(self, x):
        '''
        Forward with Gluon API.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        encoded_arr = self.encoder(x)
        self.feature_points_arr = encoded_arr
        if self.output_nn is None:
            decoded_arr = self.decoder(encoded_arr)
        else:
            inner_decoded_arr = self.output_nn(encoded_arr)
            decoded_arr = self.decoder(inner_decoded_arr)
        self.__pred_arr = decoded_arr
        return decoded_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.encoder.regularize()
        self.decoder.regularize()
        self.__tie_weights()

    def __tie_weights(self):
        if self.__tied_weights_flag is True:
            encoder_params_dict = self.encoder.extract_learned_dict()
            decoder_params_dict = self.decoder.extract_learned_dict()
            encoder_weight_keys_list = [key for key in encoder_params_dict.keys() if "hidden_units_list" in key and "weight" in key]
            decoder_weight_keys_list = [key for key in decoder_params_dict.keys() if "hidden_units_list" in key and "weight" in key]

            if len(encoder_weight_keys_list) != len(decoder_weight_keys_list):
                raise ValueError(
                    "The number of layers is invalid."
                )

            for i in range(len(self.encoder.hidden_units_list)):
                encoder_layer = i
                decoder_layer = len(self.encoder.hidden_units_list) - i - 1
                encoder_weight_keys, decoder_weight_keys = None, None
                for _encoder_weight_keys in encoder_weight_keys_list:
                    if "hidden_units_list." + str(encoder_layer) + ".weight" in _encoder_weight_keys:
                        encoder_weight_keys = _encoder_weight_keys
                        break

                for _decoder_weight_keys in decoder_weight_keys_list:
                    if "hidden_units_list." + str(decoder_layer) + ".weight" in _decoder_weight_keys:
                        decoder_weight_keys = _decoder_weight_keys
                        break

                if encoder_weight_keys is not None and decoder_weight_keys is not None:
                    try:
                        decoder_params_dict[decoder_weight_keys] = encoder_params_dict[encoder_weight_keys]
                    except AssertionError:
                        raise ValueError(
                            "The shapes of weight matrixs must be equivalents in encoder layer " + str(encoder_layer) + " and decoder layer " + str(decoder_layer)
                        )

            for k, params in decoder_params_dict.items():
                if k in decoder_weight_keys_list:
                    self.decoder.load_state_dict({k: params}, strict=False)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    def get_batch_size(self):
        ''' getter for batch size.'''
        return self.__batch_size
    
    def set_batch_size(self, value):
        ''' setter for batch size.'''
        self.__batch_size = value
    
    batch_size = property(get_batch_size, set_batch_size)

    def get_computable_loss(self):
        ''' getter for `ComputableLoss`.'''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter for `ComputableLoss`.'''
        self.__computable_loss = value
    
    computable_loss = property(get_computable_loss, set_computable_loss)
