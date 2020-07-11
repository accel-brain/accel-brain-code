# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.computable_loss import ComputableLoss

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger


class ConvolutionalNeuralNetworks(HybridBlock, ObservableData):
    '''
    Convolutional Neural Networks.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    # is-a `NNHybrid`.
    __input_nn = None
    # is-a `NNHybrid`.
    __output_nn = None

    def __init__(
        self,
        computable_loss,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        hidden_units_list=[
            Conv2D(
                channels=3,
                kernel_size=3,
                strides=(1, 1),
                padding=(1, 1),
            ), 
            Conv2D(
                channels=3,
                kernel_size=3,
                strides=(1, 1),
                padding=(1, 1),
            ),
        ],
        input_nn=None,
        input_result_height=None,
        input_result_width=None,
        input_result_channel=None,
        output_nn=None,
        hidden_dropout_rate_list=[0.5, 0.5],
        hidden_batch_norm_list=[None, None],
        optimizer_name="SGD",
        hidden_activation_list=["relu", "relu"],
        hidden_residual_flag=False,
        hidden_dense_flag=False,
        dense_axis=1,
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        not_init_flag=False,
        **kwargs
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            hidden_units_list:              `list` of `mxnet.gluon.nn._conv` in hidden layers.
            input_nn:                       is-a `NeuralNetworks` as input layers.
                                            If `None`, first layer in `hidden_units_list` will be considered as as input layer.

            output_nn:                      is-a `NeuralNetworks` as output layers.
                                            If `None`, last layer in `hidden_units_list` will be considered as an output layer.

            hidden_dropout_rate_list:       `list` of `float` of dropout rate in hidden layers.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm` in hidden layers.

            optimizer_name:                 `str` of name of optimizer.

            hidden_activation_list:         `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.

            hidden_residual_flag:           `bool` whether execute the residual learning or not in hidden layers.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:      `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        if len(hidden_units_list) != len(hidden_activation_list):
            raise ValueError("The length of `hidden_units_list` and `hidden_activation_list` must be equivalent.")

        if len(hidden_dropout_rate_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_dropout_rate_list` and `hidden_units_list` must be equivalent.")

        if len(hidden_batch_norm_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_batch_norm_list` and `hidden_units_list` must be equivalent.")

        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(ConvolutionalNeuralNetworks, self).__init__(**kwargs)

        with self.name_scope():
            if input_nn is not None:
                self.input_nn = input_nn
            if output_nn is not None:
                self.output_nn = output_nn

            self.hidden_units_list = hidden_units_list
            self.hidden_batch_norm_list = hidden_batch_norm_list
            for i in range(len(self.hidden_units_list)):
                self.register_child(self.hidden_units_list[i])

            self.hidden_dropout_rate_list = [None] * len(hidden_dropout_rate_list)
            for i in range(len(hidden_dropout_rate_list)):
                self.hidden_dropout_rate_list[i] = gluon.nn.Dropout(rate=hidden_dropout_rate_list[i])
                self.register_child(self.hidden_dropout_rate_list[i])

            for i in range(len(hidden_batch_norm_list)):
                if self.hidden_batch_norm_list[i] is not None:
                    self.register_child(self.hidden_batch_norm_list[i])

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=2
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        if self.init_deferred_flag is False:
            if not_init_flag is False:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
                if hybridize_flag is True:
                    self.hybridize()
                    if self.input_nn is not None:
                        self.input_nn.hybridize()
                    if self.output_nn is not None:
                        self.output_nn.hybridize()

        self.hidden_activation_list = hidden_activation_list

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        for v in regularizatable_data_list:
            if isinstance(v, Regularizatable) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `Regularizatable`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.input_result_height = input_result_height
        self.input_result_width = input_result_width
        self.input_result_channel = input_result_channel

        self.__hidden_residual_flag = hidden_residual_flag
        self.__hidden_dense_flag = hidden_dense_flag
        self.__dense_axis = dense_axis

        self.__ctx = ctx

        self.__safe_params_dict = {}

        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = super().collect_params(select)
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
        self.__acc_list = []
        learning_rate = self.__learning_rate
        try:
            epoch = 0
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

                # rank-3
                test_pred_arr = self.inference(test_batch_observed_arr)

                test_loss = self.compute_loss(
                    test_pred_arr,
                    test_batch_target_arr
                )

                if (epoch + 1) % 100 == 0:
                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))

                self.__loss_list.append((loss.asnumpy().mean(), test_loss.asnumpy().mean()))

                acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, batch_target_arr)
                test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_batch_target_arr)

                if ((epoch + 1) % 100 == 0):
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("Train accuracy: " + str(acc) + " Test accuracy: " + str(test_acc))
                    self.__logger.debug("Train infenreced label(inferenced):")
                    self.__logger.debug(inferenced_label_arr.asnumpy())
                    self.__logger.debug("Train infenreced label(answer):")
                    self.__logger.debug(answer_label_arr.asnumpy())

                    self.__logger.debug("Test infenreced label(inferenced):")
                    self.__logger.debug(test_inferenced_label_arr.asnumpy())
                    self.__logger.debug("Test infenreced label(answer):")
                    self.__logger.debug(test_answer_label_arr.asnumpy())
                    self.__logger.debug("-" * 100)

                    if (test_answer_label_arr[0].asnumpy() == test_answer_label_arr.asnumpy()).astype(int).sum() != test_answer_label_arr.shape[0]:
                        if (test_inferenced_label_arr[0].asnumpy() == test_inferenced_label_arr.asnumpy()).astype(int).sum() == test_inferenced_label_arr.shape[0]:
                            self.__logger.debug("It may be overfitting.")

                self.__acc_list.append(
                    (
                        acc,
                        test_acc
                    )
                )

                epoch += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def inference(self, observed_arr):
        '''
        Inference the labels.

        Args:
            observed_arr:   rank-4 Array like or sparse matrix as the observed data points.
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

    def regularize(self):
        '''
        Regularization.
        '''
        nan_flag = False
        learned_dict = self.extract_learned_dict()
        for k, params in learned_dict.items():
            if mx.nd.contrib.isnan(params).astype(int).sum() > 0:
                nan_flag = True
                break

        if nan_flag is False:
            self.__safe_params_dict = learned_dict
        else:
            if len(self.__safe_params_dict) == 0:
                raise ValueError("The vanishing gradient problem was rasied.")

            for k, params in self.collect_params().items():
                params.set_data(self.__safe_params_dict[k])

            self.__logger.debug(
                "The parameter was not updated. The vanishing gradient problem was rasied."
            )
            return

        params_dict = self.extract_learned_dict()
        for regularizatable_data in self.__regularizatable_data_list:
            params_dict = regularizatable_data.regularize(params_dict)

        for k, params in self.collect_params().items():
            params.set_data(params_dict[k])

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
        if self.input_nn is not None:
            x = self.input_nn.forward_propagation(F, x)
            x = F.reshape(
                x, 
                shape=(
                    -1, 
                    self.input_result_channel, 
                    self.input_result_height, 
                    self.input_result_width
                )
            )

        for i in range(len(self.hidden_activation_list)):
            if i > 0 and i + 1 < len(self.hidden_activation_list):
                if self.__hidden_residual_flag is True or self.__hidden_dense_flag is True:
                    _x = x

            x = self.hidden_units_list[i](x)
            if self.hidden_activation_list[i] == "identity_adjusted":
                x = x / F.sum(F.ones_like(x))
            elif self.hidden_activation_list[i] != "identity":
                x = F.Activation(x, self.hidden_activation_list[i])

            if self.hidden_dropout_rate_list[i] is not None:
                x = self.hidden_dropout_rate_list[i](x)

            if self.hidden_batch_norm_list[i] is not None:
                x = self.hidden_batch_norm_list[i](x)

            if i > 0 and i + 1 < len(self.hidden_activation_list):
                if self.__hidden_residual_flag is True:
                    x = F.elemwise_add(_x, x)
                elif self.__hidden_dense_flag is True:
                    x = F.concat(_x, x, dim=self.__dense_axis)

                if self.__hidden_residual_flag is True or self.__hidden_dense_flag is True:
                    _x = x

        if self.output_nn is not None:
            x = self.output_nn.forward_propagation(F, x)

        return x

    def compute_acc(self, prob_arr, batch_target_arr):
        '''
        Compute accuracy.

        Args:
            prob_arr:               Softmax probabilities.
            batch_target_arr:       t-hot vectors.
        
        Returns:
            Tuple data.
            - Accuracy.
            - inferenced label.
            - real label.
        '''
        inferenced_label_arr = prob_arr.argmax(axis=1)
        answer_label_arr = batch_target_arr.argmax(axis=1)
        acc = (inferenced_label_arr == answer_label_arr).sum() / self.batch_size
        return acc.asnumpy()[0], inferenced_label_arr, answer_label_arr

    def set_input_nn(self, value):
        ''' setter for `mxnet.gluon.hybridblock.HybridBlock` in input layer.'''
        if value is not None and isinstance(value, NeuralNetworks) and isinstance(value, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `input_nn` must be `NeuralNetworks`, `ConvolutionalNeuralNetworks`, or `None.")
        self.__input_nn = value
    
    def get_input_nn(self):
        ''' getter for `mxnet.gluon.hybridblock.HybridBlock` in input layer.'''
        return self.__input_nn

    input_nn = property(get_input_nn, set_input_nn)

    def set_output_nn(self, value):
        ''' setter for `mxnet.gluon.hybridblock.HybridBlock` in output layer.'''
        if value is not None and isinstance(value, NeuralNetworks) is False and isinstance(value, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `output_nn_hybird` must be `NeuralNetworks`, `ConvolutionalNeuralNetworks`, or `None.")
        self.__output_nn = value
    
    def get_output_nn(self):
        ''' getter for `mxnet.gluon.hybridblock.HybridBlock` in output layer.'''
        return self.__output_nn

    output_nn = property(get_output_nn, set_output_nn)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def set_batch_size(self, value):
        ''' ssetter for batch size. '''
        self.__batch_size = value

    def get_batch_size(self):
        ''' getter for batch size. '''
        return self.__batch_size
    
    batch_size = property(get_batch_size, set_batch_size)

    def get_loss_arr(self):
        ''' getter for for `list` of accuracies. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_acc_list(self):
        ''' getter for `list` of accuracies. '''
        return np.array(self.__acc_list)
    
    acc_arr = property(get_acc_list, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`.'''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`.'''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
