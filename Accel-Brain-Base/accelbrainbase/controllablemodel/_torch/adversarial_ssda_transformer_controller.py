# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.computableloss._torch.discriminator_loss import DiscriminatorLoss

from accelbrainbase.iteratabledata.ssda_transformer_iterator import SSDATransformerIterator

from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._torch.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.controllablemodel._torch.transformer_controller import TransformerController
import numpy as np
from logging import getLogger
import torch
from torch import nn


class AdversarialSSDATransformerController(nn.Module, ControllableModel):
    '''
    Adversarial self-supervised domain adaptive transformers.
    
    This library dryly considers the various Transformers variants 
    such as BERT, XLNet, RoBERTa, ALBERT, etc, 
    are merely applications of "self-supervised learning"
    or "self-supervised domain adaptation(SSDA)."

    From this point of view, this class builds 
    the Transformers variants as SSDA models.

    Of course, there is no necessity to build 
    an existing model like BERT or ALBERT as it is.

    This class allows you to build any combination of 
    transformers models that solve downstream tasks 
    and different models that solve pretext tasks.

    References:
        - Jing, L., & Tian, Y. (2020). Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.
        - Xu, J., Xiao, L., & López, A. M. (2019). Self-supervised domain adaptation for computer vision tasks. IEEE Access, 7, 156694-156706., p156698.

    '''
    
    def __init__(
        self,
        transformer_controller,
        downstream_task_model,
        downstream_task_loss,
        pretext_task_model_list,
        pretext_task_loss_list,
        discriminator,
        adversarial_loss=None,
        learning_rate=1e-05,
        ctx="cpu",
        scale=1.0,
        tol=3.0,
        est=1e-08,
        wd=0.0,
    ):
        '''
        Init.

        Args:
            transformer_controller:         is-a `TransformerController`.
            downstream_task_model:          is-a `NeuralNetworks` or `ConvolutionalNeuralNetworks`.
            downstream_task_loss:           is-a 
            pretext_task_model_list:        `list` of `NeuralNetworks`s or `ConvolutionalNeuralNetworks`s.
            discriminator:                  is-a `NeuralNetworks` or `ConvolutionalNeuralNetworks`.
            adversarial_loss:               is-a `ComputableLoss` or `mxnet.gluon.loss`.
                                            The arguments to be entered are as follows.
                                            - encoded feature points in target domain.
                                            - encoded feature points in source domain.
                                            In default, `DiscriminatorLoss(weight=-1.0)` as the loss function 

            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            
            optimizer_name:                 `str` of name of optimizer.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  `bool` of flag that means this class will call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            tol:                            `float` of the skipping epoch.
                                            If it is not `None`, the epoch will be skipped and parameters will be not updated 
                                            when (loss > previous_loss - tol).

            est:                            `float` of the early stopping.
                                            If it is not `None`, the learning will be stopped 
                                            when (|loss -previous_loss| < est).

        '''
        if isinstance(transformer_controller, TransformerController) is False:
            raise TypeError("The type of `transformer_controller` must be `TransformerController`.")

        if isinstance(downstream_task_model, NeuralNetworks) is False and isinstance(downstream_task_model, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `downstream_task_model` must be `NeuralNetworks` or `ConvolutionalNeuralNetworks`.")

        if isinstance(downstream_task_loss, ComputableLoss) is False and isinstance(downstream_task_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `downstream_task_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        for pretext_task_model in pretext_task_model_list:
            if isinstance(pretext_task_model, NeuralNetworks) is False and isinstance(pretext_task_model, ConvolutionalNeuralNetworks) is False:
                raise TypeError("The type of value in `pretext_task_model_list` must be `NeuralNetworks` or `ConvolutionalNeuralNetworks`.")

        for pretext_task_loss in pretext_task_loss_list:
            if isinstance(pretext_task_loss, ComputableLoss) is False and isinstance(pretext_task_loss, nn.modules.loss._Loss) is False:
                raise TypeError("The type of `pretext_task_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        if isinstance(discriminator, NeuralNetworks) is False and isinstance(discriminator, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `discriminator` must be `NeuralNetworks`.")

        if adversarial_loss is None:
            adversarial_loss = DiscriminatorLoss(weight=-1.0)

        if isinstance(adversarial_loss, ComputableLoss) is False and isinstance(adversarial_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `adversarial_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        super(AdversarialSSDATransformerController, self).__init__()

        self.transformer_controller = transformer_controller
        self.downstream_task_model = downstream_task_model
        self.downstream_task_loss = downstream_task_loss
        self.pretext_task_model_list = nn.ModuleList(pretext_task_model_list)
        self.pretext_task_loss_list = pretext_task_loss_list
        self.discriminator = discriminator
        self.adversarial_loss = adversarial_loss

        self.__learning_rate = learning_rate

        self.__ctx = ctx

        self.__loss_list = []
        self.__acc_list = []
        self.__target_domain_arr = None

        self.__tol = tol
        self.__est = est

        self.epoch = 0

    def learn(self, iteratable_data):
        '''
        Learn the observed data points with domain adaptation.

        Args:
            iteratable_data:     is-a `SSDATransformerIterator`.

        '''
        if isinstance(iteratable_data, SSDATransformerIterator) is False:
            raise TypeError("The type of `iteratable_data` must be `SSDATransformerIterator`.")

        self.__loss_list = []
        self.__acc_list = []
        learning_rate = self.__learning_rate
        self.__previous_loss = None
        est_flag = False
        try:
            epoch = self.epoch
            iter_n = 0
            for batch_data_tuple in iteratable_data.generate_learned_samples():
                (
                    encoded_observed_arr, 
                    decoded_observed_arr, 
                    encoded_mask_arr, 
                    decoded_mask_arr, 
                    test_encoded_observed_arr, 
                    test_decoded_observed_arr, 
                    test_encoded_mask_arr, 
                    test_decoded_mask_arr, 
                    training_target_arr, 
                    test_target_arr, 
                    pretext_encoded_observed_arr_list, 
                    pretext_decoded_observed_arr_list, 
                    pretext_encoded_mask_arr_list, 
                    pretext_decoded_mask_arr_list, 
                    pretext_label_arr_list
                ) = batch_data_tuple
                self.batch_size = encoded_observed_arr.shape[0]

                if epoch == 0:
                    source_encoded_arr = self.transformer_controller(
                        encoded_observed_arr, 
                        decoded_observed_arr, 
                        encoded_mask_arr, 
                        decoded_mask_arr, 
                    )
                    source_posterior_arr = self.discriminator(source_encoded_arr)
                    _ = self.downstream_task_model(source_encoded_arr)
                    target_encoded_arr_list = []
                    for i in range(len(pretext_label_arr_list)):
                        target_encoded_arr = self.transformer_controller(
                            pretext_encoded_observed_arr_list[i], 
                            pretext_decoded_observed_arr_list[i], 
                            pretext_encoded_mask_arr_list[i], 
                            pretext_decoded_mask_arr_list[i], 
                        )
                        target_encoded_arr_list.append(target_encoded_arr)

                    pred_arr = self.downstream_task_model(source_encoded_arr)
                    pretext_pred_arr_list = []
                    for i in range(len(target_encoded_arr_list)):
                        pretext_pred_arr = self.pretext_task_model_list[i](
                            target_encoded_arr_list[i]
                        )
                        pretext_pred_arr_list.append(pretext_pred_arr)

                self.transformer_controller.optimizer.zero_grad()
                self.transformer_controller.encoder.optimizer.zero_grad()
                self.transformer_controller.decoder.optimizer.zero_grad()
                self.discriminator.optimizer.zero_grad()
                self.downstream_task_model.optimizer.zero_grad()
                for i in range(len(self.pretext_task_model_list)):
                    self.pretext_task_model_list[i].optimizer.zero_grad()

                source_encoded_arr = self.transformer_controller(
                    encoded_observed_arr, 
                    decoded_observed_arr, 
                    encoded_mask_arr, 
                    decoded_mask_arr, 
                )
                source_posterior_arr = self.discriminator(source_encoded_arr)

                target_encoded_arr_list = []
                adversarial_loss = 0.0
                for i in range(len(pretext_encoded_observed_arr_list)):
                    target_encoded_arr = self.transformer_controller(
                        pretext_encoded_observed_arr_list[i], 
                        pretext_decoded_observed_arr_list[i], 
                        pretext_encoded_mask_arr_list[i], 
                        pretext_decoded_mask_arr_list[i], 
                    )
                    target_encoded_arr_list.append(target_encoded_arr)

                    target_posterior_arr = self.discriminator(target_encoded_arr)
                    _adversarial_loss = self.adversarial_loss(
                        target_posterior_arr,
                        source_posterior_arr
                    )
                    adversarial_loss = adversarial_loss + _adversarial_loss

                pred_arr = self.downstream_task_model(source_encoded_arr)

                pretext_pred_arr_list = []
                for i in range(len(target_encoded_arr_list)):
                    pretext_pred_arr = self.pretext_task_model_list[i](
                        target_encoded_arr_list[i]
                    )
                    pretext_pred_arr_list.append(pretext_pred_arr)

                pretext_loss = 0.0
                for i in range(len(self.pretext_task_loss_list)):
                    pretext_loss = pretext_loss + self.pretext_task_loss_list[i](
                        pretext_pred_arr_list[i],
                        pretext_label_arr_list[i]
                    )

                downstream_task_loss = self.downstream_task_loss(
                    pred_arr,
                    training_target_arr
                )

                loss = downstream_task_loss + pretext_loss + adversarial_loss

                if self.__previous_loss is not None:
                    loss_diff = loss - self.__previous_loss
                    loss_diff = loss_diff.to('cpu').detach().numpy().copy()
                    if self.__est is not None and np.abs(loss_diff) < self.__est:
                        est_flag = True

                if est_flag is False:
                    loss.backward()
                    for i in range(len(self.pretext_task_model_list)):
                        self.pretext_task_model_list[i].optimizer.step()
                    self.downstream_task_model.optimizer.step()
                    self.discriminator.optimizer.step()
                    self.transformer_controller.optimizer.step()
                    self.transformer_controller.decoder.optimizer.step()
                    self.transformer_controller.encoder.optimizer.step()
                    self.regularize()

                    self.__previous_loss = loss

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        with torch.inference_mode():
                            source_encoded_arr = self.transformer_controller(
                                test_encoded_observed_arr, 
                                test_decoded_observed_arr, 
                                test_encoded_mask_arr, 
                                test_decoded_mask_arr, 
                            )
                            source_posterior_arr = self.discriminator(source_encoded_arr)
                            target_encoded_arr_list = []
                            test_adversarial_loss = 0.0
                            for i in range(len(pretext_encoded_observed_arr_list)):
                                target_encoded_arr = self.transformer_controller(
                                    pretext_encoded_observed_arr_list[i], 
                                    pretext_decoded_observed_arr_list[i], 
                                    pretext_encoded_mask_arr_list[i], 
                                    pretext_decoded_mask_arr_list[i], 
                                )
                                target_encoded_arr_list.append(target_encoded_arr)

                                target_posterior_arr = self.discriminator(target_encoded_arr)
                                _test_adversarial_loss = self.adversarial_loss(
                                    target_posterior_arr,
                                    source_posterior_arr
                                )
                                test_adversarial_loss = test_adversarial_loss + _test_adversarial_loss

                            test_pred_arr = self.downstream_task_model(source_encoded_arr)
                            pretext_pred_arr_list = []
                            for i in range(len(target_encoded_arr_list)):
                                pretext_pred_arr = self.pretext_task_model_list[i](
                                    target_encoded_arr_list[i]
                                )
                                pretext_pred_arr_list.append(pretext_pred_arr)

                            test_pretext_loss = 0.0
                            for i in range(len(self.pretext_task_loss_list)):
                                test_pretext_loss = pretext_loss + self.pretext_task_loss_list[i](
                                    pretext_pred_arr_list[i],
                                    pretext_label_arr_list[i]
                                )

                            test_downstream_task_loss = self.downstream_task_loss(
                                test_pred_arr,
                                test_target_arr
                            )

                            test_loss = test_downstream_task_loss + test_pretext_loss + test_adversarial_loss

                        _loss = loss.to('cpu').detach().numpy().copy()
                        _downstream_task_loss = downstream_task_loss.to('cpu').detach().numpy().copy()
                        _pretext_loss = pretext_loss.to('cpu').detach().numpy().copy()
                        _adversarial_loss = adversarial_loss.to("cpu").detach().numpy().copy()
                        _test_loss = test_loss.to('cpu').detach().numpy().copy()
                        _test_downstream_task_loss = test_downstream_task_loss.to('cpu').detach().numpy().copy()
                        _test_pretext_loss = test_pretext_loss.to('cpu').detach().numpy().copy()
                        _test_adversarial_loss = test_adversarial_loss.to("cpu").detach().numpy().copy()

                        self.__logger.debug("Epochs: " + str(epoch + 1) + " Train total loss: " + str(_loss) + " Test total loss: " + str(_test_loss))
                        self.__logger.debug("Train classification loss: " + str(_downstream_task_loss) + " Test classification loss: " + str(_test_downstream_task_loss))
                        self.__logger.debug("Train Pretext task loss: " + str(_pretext_loss) + " Test Pretext task loss: " + str(_test_pretext_loss))
                        self.__logger.debug("Train Adversarial loss: " + str(_adversarial_loss) + " Test Adversarial loss: " + str(_test_adversarial_loss))

                    if self.compute_acc_flag is True:
                        if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                            acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, training_target_arr)
                            test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_target_arr)
                            if (epoch + 1) % 100 == 0 or epoch < 100:
                                acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, training_target_arr)
                                test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_target_arr)

                                self.__logger.debug("-" * 100)
                                self.__logger.debug("Train accuracy: " + str(acc) + " Test accuracy: " + str(test_acc))
                                self.__logger.debug("Train infenreced label(inferenced):")
                                self.__logger.debug(inferenced_label_arr)
                                self.__logger.debug("Train infenreced label(answer):")
                                self.__logger.debug(answer_label_arr)

                                self.__logger.debug("Test infenreced label(inferenced):")
                                self.__logger.debug(test_inferenced_label_arr)
                                self.__logger.debug("Test infenreced label(answer):")
                                self.__logger.debug(test_answer_label_arr)
                                self.__logger.debug("-" * 100)

                                if (test_answer_label_arr[0] == test_answer_label_arr).astype(int).sum() != test_answer_label_arr.shape[0]:
                                    if (test_inferenced_label_arr[0] == test_inferenced_label_arr).astype(int).sum() == test_inferenced_label_arr.shape[0]:
                                        self.__logger.debug("It may be overfitting.")

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        self.__loss_list.append(
                            (
                                _loss, 
                                _test_loss,
                                _downstream_task_loss,
                                _test_downstream_task_loss,
                                _pretext_loss,
                                _test_pretext_loss,
                                _adversarial_loss,
                                _test_adversarial_loss,
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
                    self.__logger.debug(str(self.__previous_loss) + " -> " + str(loss_mean))
                    break

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")
        self.epoch = epoch

    def inference(
        self, 
        encoded_observed_arr, 
        decoded_observed_arr, 
        encoded_mask_arr=None, 
        decoded_mask_arr=None, 
    ):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(
            encoded_observed_arr, 
            decoded_observed_arr, 
            encoded_mask_arr, 
            decoded_mask_arr, 
        )

    def forward(
        self, 
        encoded_observed_arr, 
        decoded_observed_arr, 
        encoded_mask_arr, 
        decoded_mask_arr, 
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        source_encoded_arr = self.transformer_controller(
            encoded_observed_arr, 
            decoded_observed_arr, 
            encoded_mask_arr, 
            decoded_mask_arr, 
        )
        pred_arr = self.downstream_task_model(source_encoded_arr)
        return pred_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.transformer_controller.regularize()
        self.downstream_task_model.regularize()
        for i in range(len(self.pretext_task_model_list)):
            self.pretext_task_model_list[i].regularize()

    def compute_loss(
        self, 
        pred_arr, 
        label_arr, 
    ):
        '''
        Compute loss.

        Args:
            pretext_pred_arr:       `mxnet.ndarray` or `mxnet.symbol` of predicted data in pretext, or target domain.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points in source domain.
            pretext_label_arr:      `mxnet.ndarray` or `mxnet.symbol` of label data in pretext.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data in source domain.

        Returns:
            Tensor of losses.
        '''
        return self.downstream_task_loss(
            pred_arr, 
            label_arr, 
        )

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
        inferenced_label_arr = prob_arr.argmax(axis=1).to('cpu').detach().numpy().copy()
        answer_label_arr = batch_target_arr.argmax(axis=1).to('cpu').detach().numpy().copy()
        acc = (inferenced_label_arr == answer_label_arr).astype(int).sum() / self.batch_size
        return acc, inferenced_label_arr, answer_label_arr

    # `bool`, compute accuracy or not in learning.
    __compute_acc_flag = True

    def get_compute_acc_flag(self):
        ''' getter '''
        return self.__compute_acc_flag
    
    def set_compute_acc_flag(self, value):
        ''' setter '''
        self.__compute_acc_flag = value

    compute_acc_flag = property(get_compute_acc_flag, set_compute_acc_flag)

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
