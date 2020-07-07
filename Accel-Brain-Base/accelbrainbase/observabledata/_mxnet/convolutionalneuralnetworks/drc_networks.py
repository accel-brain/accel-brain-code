# -*- coding: utf-8 -*-
from accelbrainbase.iteratabledata._mxnet.drcn_iterator import DRCNIterator
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.computableloss._mxnet.drcn_loss import DRCNLoss
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class DRCNetworks(ConvolutionalNeuralNetworks):
    '''
    Deep Reconstruction-Classification Networks(DRCN or DRCNetworks).

    Deep Reconstruction-Classification Network(DRCN or DRCNetworks) is a convolutional network 
    that jointly learns two tasks: 
    
    1. supervised source label prediction.
    2. unsupervised target data reconstruction. 

    Ideally, a discriminative representation should model both the label and 
    the structure of the data. Based on that intuition, Ghifary, M., et al.(2016) hypothesize 
    that a domain-adaptive representation should satisfy two criteria:
    
    1. classify well the source domain labeled data.
    2. reconstruct well the target domain unlabeled data, which can be viewed as an approximate of the ideal discriminative representation.

    The encoding parameters of the DRCN are shared across both tasks, 
    while the decoding parameters are sepa-rated. The aim is that the learned label 
    prediction function can perform well onclassifying images in the target domain
    thus the data reconstruction can beviewed as an auxiliary task to support the 
    adaptation of the label prediction.

    References:
        - Ghifary, M., Kleijn, W. B., Zhang, M., Balduzzi, D., & Li, W. (2016, October). Deep reconstruction-classification networks for unsupervised domain adaptation. In European Conference on Computer Vision (pp. 597-613). Springer, Cham.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    # `list` of losses.
    __loss_list = []

    # `list` of accuracies.
    __acc_list = []

    def __init__(
        self,
        convolutional_auto_encoder,
        drcn_loss,
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
        tol=3.0,
        est=1e-08,
        wd=0.0,
        **kwargs
    ):
        '''
        Init.

        Args:
            convolutional_auto_encoder:     is-a `ConvolutionalAutoEncoder`.
            drcn_loss:               is-a `DRCNLoss`.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            

            hidden_units_list:              `list` of `mxnet.gluon.nn._conv` in hidden layers.
            output_nn:                      is-a `NeuralNetworks` as output layers.
                                            If `None`, last layer in `hidden_units_list` will be considered as an output layer.

            hidden_dropout_rate_list:       `list` of `float` of dropout rate in hidden layers.

            optimizer_name:                 `str` of name of optimizer.

            hidden_activation_list:         `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm` in hidden layers.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  `bool` of flag that means this class will call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:           `list` of `Regularizatable`.
            scale:                          `float` of scaling factor for initial parameters.
            tied_weights_flag:              `bool` of flag to tied weights or not.
            tol:                            `float` of the skipping epoch.
                                            If it is not `None`, the epoch will be skipped and parameters will be not updated 
                                            when (loss > previous_loss - tol).

            est:                            `float` of the early stopping.
                                            If it is not `None`, the learning will be stopped 
                                            when (|loss -previous_loss| < est).

        '''
        if isinstance(convolutional_auto_encoder, ConvolutionalAutoEncoder) is False:
            raise TypeError("The type of `convolutional_auto_encoder` must be `ConvolutionalAutoEncoder`.")

        if len(hidden_units_list) != len(hidden_activation_list):
            raise ValueError("The length of `hidden_units_list` and `hidden_activation_list` must be equivalent.")

        if len(hidden_dropout_rate_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_dropout_rate_list` and `hidden_units_list` must be equivalent.")

        if isinstance(drcn_loss, DRCNLoss) is False:
            raise TypeError("The type of `drcn_loss` must be `DRCNLoss`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

        super().__init__(
            computable_loss=drcn_loss,
            initializer=initializer,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            hidden_units_list=hidden_units_list,
            output_nn=None,
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
        self.convolutional_auto_encoder = convolutional_auto_encoder
        self.__tied_weights_flag = tied_weights_flag

        self.output_nn = None

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
                self.collect_params().initialize(
                    self.initializer,
                    force_reinit=True, 
                    ctx=ctx
                )
                self.trainer = gluon.Trainer(
                    self.collect_params(), 
                    optimizer_name, 
                    {
                        "learning_rate": learning_rate,
                        "wd": wd,
                    }
                )

                if output_nn is not None:
                    output_nn.collect_params().initialize(
                        output_nn.initializer,
                        force_reinit=True,
                        ctx=ctx
                    )
                    self.output_nn_trainer = gluon.Trainer(
                        output_nn.collect_params(),
                        optimizer_name,
                        {
                            "learning_rate": learning_rate,
                            "wd": wd,
                        }
                    )
                else:
                    self.output_nn_trainer = None

                self.output_nn = output_nn

                if hybridize_flag is True:
                    self.convolutional_auto_encoder.encoder.hybridize()
                    self.convolutional_auto_encoder.decoder.hybridize()

                    if self.output_nn is not None:
                        self.output_nn.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        for v in regularizatable_data_list:
            if isinstance(v, Regularizatable) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `Regularizatable`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.drcn_loss = drcn_loss

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__ctx = ctx

        self.__loss_list = []
        self.__acc_list = []
        self.__target_domain_arr = None

        self.__tied_weights_flag = tied_weights_flag

        self.__tol = tol
        self.__est = est

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = super().collect_params(select)
        params_dict.update(self.convolutional_auto_encoder.collect_params(select))

        if self.output_nn is not None:
            params_dict.update(self.output_nn.collect_params(select))

        return params_dict

    def learn(self, iteratable_data):
        '''
        Learn the observed data points with domain adaptation.

        Args:
            iteratable_data:     is-a `DRCNIterator`.

        '''
        if isinstance(iteratable_data, DRCNIterator) is False:
            raise TypeError("The type of `iteratable_data` must be `DRCNIterator`.")

        self.__loss_list = []
        self.__acc_list = []
        learning_rate = self.__learning_rate
        self.__previous_loss = None
        tol_flag = False
        est_flag = False
        try:
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr, target_domain_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                self.batch_size = batch_observed_arr.shape[0]
                self.convolutional_auto_encoder.batch_size = self.batch_size
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)
                    if self.output_nn_trainer is not None:
                        self.output_nn_trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    # rank-3
                    decoded_arr = self.inference_auto_encoder(target_domain_arr)
                    _, prob_arr = self.inference(batch_observed_arr)
                    loss, train_classification_loss, train_reconstruction_loss = self.compute_loss(
                        decoded_arr, 
                        prob_arr,
                        target_domain_arr,
                        batch_target_arr
                    )

                loss_mean = loss.mean().asscalar()
                if self.__previous_loss is not None:
                    loss_diff = loss_mean - self.__previous_loss
                    if self.__tol is not None and loss_diff > self.__tol:
                        tol_flag = True
                    if self.__est is not None and np.abs(loss_diff) < self.__est:
                        est_flag = True

                if est_flag is False:
                    if tol_flag is True:
                        if ((epoch + 1) % 100 == 0):
                            self.__logger.debug("TOL...")
                            self.__logger.debug(str(self.__previous_loss) + " -> " + str(loss_mean))
                        self.trainer.set_learning_rate(0.0)
                        if self.output_nn_trainer is not None:
                            self.output_nn_trainer.set_learning_rate(0.0)

                    loss.backward()
                    self.trainer.step(batch_observed_arr.shape[0])
                    if self.output_nn_trainer is not None:
                        self.output_nn_trainer.step(batch_observed_arr.shape[0])
                    self.regularize()

                    self.__previous_loss = loss_mean

                    # rank-3
                    test_decoded_arr = self.inference_auto_encoder(target_domain_arr)
                    _, test_prob_arr = self.inference(test_batch_observed_arr)
                    test_loss, test_classification_loss, test_reconstruction_loss = self.compute_loss(
                        test_decoded_arr, 
                        test_prob_arr,
                        target_domain_arr,
                        test_batch_target_arr
                    )

                    if (epoch + 1) % 100 == 0:
                        self.__logger.debug("Epochs: " + str(epoch + 1) + " Train total loss: " + str(loss.asnumpy().mean()) + " Test total loss: " + str(test_loss.asnumpy().mean()))
                        self.__logger.debug("Train classification loss: " + str(train_classification_loss.asnumpy().mean()) + " Test classification loss: " + str(test_classification_loss.asnumpy().mean()))
                        self.__logger.debug("Train reconstruction loss: " + str(train_reconstruction_loss.asnumpy().mean()) + " Test reconstruction loss: " + str(test_reconstruction_loss.asnumpy().mean()))

                    acc, inferenced_label_arr, answer_label_arr = self.compute_acc(prob_arr, batch_target_arr)
                    test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_prob_arr, test_batch_target_arr)

                    if ((epoch + 1) % 100 == 0):
                        acc, inferenced_label_arr, answer_label_arr = self.compute_acc(prob_arr, batch_target_arr)
                        test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_prob_arr, test_batch_target_arr)

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

                    self.__loss_list.append(
                        (
                            loss.asnumpy().mean(), 
                            test_loss.asnumpy().mean(),
                            train_classification_loss.asnumpy().mean(),
                            test_classification_loss.asnumpy().mean(),
                            train_reconstruction_loss.asnumpy().mean(),
                            test_reconstruction_loss.asnumpy().mean()
                        )
                    )
                    self.__acc_list.append(
                        (
                            acc,
                            test_acc
                        )
                    )
                    epoch += 1

                    if tol_flag is True:
                        self.trainer.set_learning_rate(learning_rate)
                        if self.output_nn_trainer is not None:
                            self.output_nn_trainer.set_learning_rate(learning_rate)

                else:
                    self.__logger.debug("Early stopping.")
                    self.__logger.debug(str(self.__previous_loss) + " -> " + str(loss_mean))
                    break

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            Tuple data.
                - `mxnet.ndarray` or `mxnet.symbol` of reconstrcted feature points.
                - `mxnet.ndarray` or `mxnet.symbol` of inferenced label.
        '''
        decoded_arr = self.convolutional_auto_encoder.forward_propagation(F, x)
        self.feature_points_arr = self.convolutional_auto_encoder.feature_points_arr

        if self.output_nn is not None:
            prob_arr = self.output_nn.forward_propagation(F, self.feature_points_arr)
        else:
            prob_arr = self.feature_points_arr

        return decoded_arr, prob_arr

    def inference_auto_encoder(self, x):
        '''
        Hybrid forward with Gluon API (Auto-Encoder only).

        Args:
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of reconstrcted feature points.
        '''
        return self.convolutional_auto_encoder.inference(x)

    def regularize(self):
        '''
        Regularization.
        '''
        if self.__tied_weights_flag is True:
            self.convolutional_auto_encoder.tie_weights()
        super().regularize()

    def compute_loss(
        self, 
        decoded_arr, 
        prob_arr,
        batch_observed_arr,
        batch_target_arr
    ):
        '''
        Compute loss.

        Args:
            decoded_arr:            `mxnet.ndarray` or `mxnet.symbol` of decoded feature points..
            prob_arr:               `mxnet.ndarray` or `mxnet.symbol` of predicted labels data.
            batch_observed_arr:     `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            batch_target_arr:       `mxnet.ndarray` or `mxnet.symbol` of label data.

        Returns:
            loss.
        '''
        return self.drcn_loss(
            decoded_arr, 
            prob_arr,
            batch_observed_arr,
            batch_target_arr
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_acc_list(self):
        ''' getter for accuracies. '''
        return np.array(self.__acc_list)
    
    acc_arr = property(get_acc_list, set_readonly)
