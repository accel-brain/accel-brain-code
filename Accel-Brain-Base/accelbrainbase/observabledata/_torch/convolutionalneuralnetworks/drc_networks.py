# -*- coding: utf-8 -*-
from accelbrainbase.iteratabledata._torch.drcn_iterator import DRCNIterator
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.computableloss._torch.drcn_loss import DRCNLoss
from accelbrainbase.observabledata._torch.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._torch.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder

import torch
from torch import nn
from torch.optim.adam import Adam

import numpy as np
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
        initializer_f=None,
        optimizer_f=None,
        auto_encoder_optimizer_f=None,
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
        est=1e-08,
        wd=0.0,
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            convolutional_auto_encoder:     is-a `ConvolutionalAutoEncoder`.
            drcn_loss:                      is-a `DRCNLoss`.
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
            initializer_f=initializer_f,
            learning_rate=learning_rate,
            hidden_units_list=hidden_units_list,
            output_nn=None,
            hidden_dropout_rate_list=hidden_dropout_rate_list,
            optimizer_f=optimizer_f,
            hidden_activation_list=hidden_activation_list,
            hidden_batch_norm_list=hidden_batch_norm_list,
            ctx=ctx,
            regularizatable_data_list=regularizatable_data_list,
            scale=scale,
        )
        self.init_deferred_flag = init_deferred_flag
        self.convolutional_auto_encoder = convolutional_auto_encoder
        self.__tied_weights_flag = tied_weights_flag

        self.output_nn = output_nn

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.drcn_loss = drcn_loss

        self.__learning_rate = learning_rate

        self.__ctx = ctx

        self.__loss_list = []
        self.__acc_list = []
        self.__target_domain_arr = None

        self.__tied_weights_flag = tied_weights_flag

        self.__est = est

        self.__not_init_flag = not_init_flag

        for i in range(len(hidden_units_list)):
            if initializer_f is None:
                hidden_units_list[i].weight = torch.nn.init.xavier_normal_(
                    hidden_units_list[i].weight,
                    gain=1.0
                )
            else:
                hidden_units_list[i].weight = initializer_f(hidden_units_list[i].weight)

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if auto_encoder_optimizer_f is not None:
                    self.convolutional_auto_encoder.encoder_optimizer = auto_encoder_optimizer_f(
                        self.convolutional_auto_encoder.encoder.parameters(), 
                    )
                    self.convolutional_auto_encoder.decoder_optimizer = auto_encoder_optimizer_f(
                        self.convolutional_auto_encoder.decoder.parameters(), 
                    )
                elif optimizer_f is not None:
                    self.convolutional_auto_encoder.encoder_optimizer = optimizer_f(
                        self.convolutional_auto_encoder.encoder.parameters(), 
                    )
                    self.convolutional_auto_encoder.decoder_optimizer = optimizer_f(
                        self.convolutional_auto_encoder.decoder.parameters(), 
                    )
                else:
                    self.convolutional_auto_encoder.encoder_optimizer = Adam(
                        self.convolutional_auto_encoder.encoder.parameters(),
                        lr=self.__learning_rate
                    )
                    self.convolutional_auto_encoder.decoder_optimizer = Adam(
                        self.convolutional_auto_encoder.decoder.parameters(),
                        lr=self.__learning_rate
                    )

                if optimizer_f is not None:
                    self.optimizer = optimizer_f(
                        self.output_nn.parameters(),
                        lr=self.__learning_rate,
                    )
                elif optimizer_f is not None:
                    self.optimizer = optimizer_f(
                        self.parameters(), 
                    )
                else:
                    self.optimizer = Adam(
                        self.parameters(), 
                        lr=self.__learning_rate,
                    )

        self.flatten = nn.Flatten()

    def parameters(self):
        '''
        '''
        params_dict_list = [
            {
                "params": self.convolutional_auto_encoder.parameters(),
            },
            {
                "params": self.parameters(),
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
        est_flag = False
        try:
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr, target_domain_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                self.batch_size = batch_observed_arr.shape[0]
                self.convolutional_auto_encoder.batch_size = self.batch_size

                self.convolutional_auto_encoder.encoder_optimizer.zero_grad()
                self.convolutional_auto_encoder.decoder_optimizer.zero_grad()
                self.optimizer.zero_grad()

                # rank-3
                decoded_arr = self.inference_auto_encoder(target_domain_arr)
                _, prob_arr = self.inference(batch_observed_arr)
                loss, train_classification_loss, train_reconstruction_loss = self.compute_loss(
                    decoded_arr, 
                    prob_arr,
                    target_domain_arr,
                    batch_target_arr
                )

                if self.__previous_loss is not None:
                    loss_diff = loss - self.__previous_loss
                    if self.__est is not None and torch.abs(loss_diff) < self.__est:
                        est_flag = True

                if est_flag is False:
                    loss.backward()
                    self.convolutional_auto_encoder.encoder_optimizer.step()
                    self.convolutional_auto_encoder.decoder_optimizer.step()
                    self.optimizer.step()
                    self.regularize()

                    self.__previous_loss = loss

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        with torch.inference_mode():
                            # rank-3
                            test_decoded_arr = self.inference_auto_encoder(target_domain_arr)
                            _, test_prob_arr = self.inference(test_batch_observed_arr)
                            test_loss, test_classification_loss, test_reconstruction_loss = self.compute_loss(
                                test_decoded_arr, 
                                test_prob_arr,
                                target_domain_arr,
                                test_batch_target_arr
                            )

                        _loss = loss.to('cpu').detach().numpy().copy()
                        _train_classification_loss = train_classification_loss.to('cpu').detach().numpy().copy()
                        _train_reconstruction_loss = train_reconstruction_loss.to('cpu').detach().numpy().copy()
                        _test_loss = test_loss.to('cpu').detach().numpy().copy()
                        _test_classification_loss = test_classification_loss.to('cpu').detach().numpy().copy()
                        _test_reconstruction_loss = test_reconstruction_loss.to('cpu').detach().numpy().copy()

                        self.__logger.debug("Epochs: " + str(epoch + 1) + " Train total loss: " + str(_loss) + " Test total loss: " + str(_test_loss))
                        self.__logger.debug("Train classification loss: " + str(_train_classification_loss) + " Test classification loss: " + str(_test_classification_loss))
                        self.__logger.debug("Train reconstruction loss: " + str(_train_reconstruction_loss) + " Test reconstruction loss: " + str(_test_reconstruction_loss))

                    if self.compute_acc_flag is True:
                        if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                            acc, inferenced_label_arr, answer_label_arr = self.compute_acc(prob_arr, batch_target_arr)
                            test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_prob_arr, test_batch_target_arr)
                            if (epoch + 1) % 100 == 0 or epoch < 100:
                                acc, inferenced_label_arr, answer_label_arr = self.compute_acc(prob_arr, batch_target_arr)
                                test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_prob_arr, test_batch_target_arr)

                                self.__logger.debug("-" * 100)
                                self.__logger.debug("Train accuracy: " + str(acc) + " Test accuracy: " + str(test_acc))
                                self.__logger.debug("Train infenreced label(inferenced):")
                                self.__logger.debug(inferenced_label_arr.to('cpu').detach().numpy())
                                self.__logger.debug("Train infenreced label(answer):")
                                self.__logger.debug(answer_label_arr.to('cpu').detach().numpy())

                                self.__logger.debug("Test infenreced label(inferenced):")
                                self.__logger.debug(test_inferenced_label_arr.to('cpu').detach().numpy())
                                self.__logger.debug("Test infenreced label(answer):")
                                self.__logger.debug(test_answer_label_arr.to('cpu').detach().numpy())
                                self.__logger.debug("-" * 100)

                                if (test_answer_label_arr[0] == test_answer_label_arr).to('cpu').detach().numpy().astype(int).sum() != test_answer_label_arr.shape[0]:
                                    if (test_inferenced_label_arr[0] == test_inferenced_label_arr).to('cpu').detach().numpy().astype(int).sum() == test_inferenced_label_arr.shape[0]:
                                        self.__logger.debug("It may be overfitting.")

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        self.__loss_list.append(
                            (
                                _loss, 
                                _test_loss,
                                _train_classification_loss,
                                _test_classification_loss,
                                _train_reconstruction_loss,
                                _test_reconstruction_loss
                            )
                        )
                        if self.compute_acc_flag is True:
                            self.__acc_list.append(
                                (
                                    acc,
                                    test_acc
                                )
                            )

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        epoch += 1
                    iter_n += 1

                else:
                    self.__logger.debug("Early stopping.")
                    break

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            Tuple data.
                - `tensor` of reconstrcted feature points.
                - `tensor` of inferenced label.
        '''
        decoded_arr = self.convolutional_auto_encoder(x)
        self.feature_points_arr = self.convolutional_auto_encoder.feature_points_arr

        if self.output_nn is not None:
            prob_arr = self.output_nn(self.feature_points_arr)
        else:
            prob_arr = self.feature_points_arr

        return decoded_arr, prob_arr

    def inference_auto_encoder(self, x):
        '''
        Hybrid forward with Gluon API (Auto-Encoder only).

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of reconstrcted feature points.
        '''
        return self.convolutional_auto_encoder(x)

    def regularize(self):
        '''
        Regularization.
        '''
        self.convolutional_auto_encoder.regularize()
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
            decoded_arr:            `tensor` of decoded feature points..
            prob_arr:               `tensor` of predicted labels data.
            batch_observed_arr:     `tensor` of observed data points.
            batch_target_arr:       `tensor` of label data.

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
