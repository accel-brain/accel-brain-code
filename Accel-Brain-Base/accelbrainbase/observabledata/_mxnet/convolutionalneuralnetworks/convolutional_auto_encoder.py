# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
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
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        hidden_units_list=[],
        output_nn=None,
        hidden_dropout_rate_list=[],
        optimizer_name="SGD",
        hidden_activation_list=[],
        hidden_batch_norm_list=[],
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        tied_weights_flag=True,
        init_deferred_flag=None,
        wd=None,
        **kwargs
    ):
        '''
        Init.

        Args:
            encoder:                        is-a `CNNHybrid`.
            decoder:                        is-a `CNNHybrid`.
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            

            hidden_units_list:              `list` of `mxnet.gluon.nn._conv` in hidden layers.
            output_nn:                      is-a `NNHybrid` as output layers.
                                            If `None`, last layer in `hidden_units_list` will be considered as an output layer.

            hidden_dropout_rate_list:       `list` of `float` of dropout rate in hidden layers.

            optimizer_name:                 `str` of name of optimizer.

            hidden_activation_list:         `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm` in hidden layers.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
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

        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        if init_deferred_flag is None:
            init_deferred_flag = self.init_deferred_flag
        elif isinstance(init_deferred_flag, bool) is False:
            raise TypeError("The type of `init_deferred_flag` must be `bool`.")

        self.init_deferred_flag = True

        super().__init__(
            computable_loss=computable_loss,
            initializer=initializer,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            hidden_units_list=hidden_units_list,
            output_nn=output_nn,
            hidden_dropout_rate_list=hidden_dropout_rate_list,
            optimizer_name=optimizer_name,
            hidden_activation_list=hidden_activation_list,
            hidden_batch_norm_list=hidden_batch_norm_list,
            ctx=ctx,
            hybridize_flag=hybridize_flag,
            regularizatable_data_list=regularizatable_data_list,
            scale=scale,
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag
        self.encoder = encoder
        self.decoder = decoder
        self.__tied_weights_flag = tied_weights_flag
        self.output_nn = output_nn

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=1
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

            self.initializer = initializer

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                params_dict = {
                    "learning_rate": learning_rate
                }
                if wd is not None:
                    params_dict.setdefault("wd", wd)

                self.trainer = gluon.Trainer(
                    self.collect_params(), 
                    optimizer_name, 
                    params_dict
                )
                if hybridize_flag is True:
                    self.encoder.hybridize()
                    self.decoder.hybridize()
                    if self.output_nn is not None:
                        self.output_nn.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        self.__computable_loss = computable_loss

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__ctx = ctx

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.encoder.collect_params(select)
        params_dict.update(self.decoder.collect_params(select))
        if self.output_nn is not None:
            params_dict.update(self.output_nn.collect_params(select))
        return params_dict

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
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    # rank-3
                    pred_arr = self.inference(batch_observed_arr)
                    loss = self.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                loss.backward()
                self.trainer.step(batch_observed_arr.shape[0])
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    # rank-3
                    test_pred_arr = self.inference(test_batch_observed_arr)
                    test_loss = self.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    self.__loss_list.append((loss.asnumpy().mean(), test_loss.asnumpy().mean()))

                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))
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
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr)

    def compute_loss(self, pred_arr, labeled_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       `mxnet.ndarray` or `mxnet.symbol`.
            labeled_arr:    `mxnet.ndarray` or `mxnet.symbol`.

        Returns:
            loss.
        '''
        return self.__computable_loss(pred_arr, labeled_arr)

    def extract_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it.

        Returns:
            The `mxnet.ndarray` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return self.feature_points_arr

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = self.collect_params()
        
        params_arr_dict = {}
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        encoded_arr = self.encoder.forward_propagation(F, x)
        self.feature_points_arr = encoded_arr
        if self.output_nn is None:
            decoded_arr = self.decoder.forward_propagation(F, encoded_arr)
        else:
            inner_decoded_arr = self.output_nn.forward_propagation(F, encoded_arr)
            decoded_arr = self.decoder.forward_propagation(F, inner_decoded_arr)
        self.__pred_arr = decoded_arr
        return decoded_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.tie_weights()
        super().regularize()

    def tie_weights(self):
        '''
        Tie weights.
        '''
        if self.__tied_weights_flag is True:
            encoder_params_dict = self.encoder.extract_learned_dict()
            decoder_params_dict = self.decoder.extract_learned_dict()
            decoder_weight_keys_list = []
            for i in range(len(self.encoder.hidden_units_list)):
                encoder_layer = i
                decoder_layer = len(self.encoder.hidden_units_list) - i - 1
                encoder_weight_key = self.encoder.hidden_units_list[encoder_layer].name + "_weight"
                decoder_weight_key = self.decoder.hidden_units_list[decoder_layer].name + "_weight"

                if encoder_weight_key not in encoder_params_dict or decoder_weight_key not in decoder_params_dict:
                    continue

                try:
                    if decoder_params_dict[decoder_weight_key].shape != encoder_params_dict[encoder_weight_key].shape:
                        raise AssertionError()

                    decoder_params_dict[decoder_weight_key] = encoder_params_dict[encoder_weight_key]
                except AssertionError:
                    # TypeError ?
                    raise ValueError(
                        "The shapes of weight matrixs must be equivalents in encoder layer " + str(encoder_layer) + " and decoder layer " + str(decoder_layer)
                    )
                decoder_weight_keys_list.append(decoder_weight_key)

            for k, params in self.decoder.collect_params().items():
                if k in decoder_weight_keys_list:
                    params.set_data(decoder_params_dict[k])

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
