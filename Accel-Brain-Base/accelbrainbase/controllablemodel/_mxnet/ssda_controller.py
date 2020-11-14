# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel

from accelbrainbase.iteratabledata._mxnet.ssda_iterator import SSDAIterator
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computableloss._mxnet.ssda_loss import SSDALoss
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger
from mxnet.gluon.block import HybridBlock


class SSDAController(HybridBlock, ControllableModel):
    '''
    Self-supervised domain adaptation.

    References:
        - Jing, L., & Tian, Y. (2020). Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.
        - Xu, J., Xiao, L., & López, A. M. (2019). Self-supervised domain adaptation for computer vision tasks. IEEE Access, 7, 156694-156706., p156698.

    '''
    
    def __init__(
        self,
        encoder,
        classifier,
        pretext_task_model,
        ssda_loss,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        optimizer_name="SGD",
        ctx=mx.gpu(),
        hybridize_flag=True,
        scale=1.0,
        tol=3.0,
        est=1e-08,
        wd=0.0,
        **kwargs
    ):
        '''
        Init.

        Args:
            encoder:                        is-a `ConvolutionalNeuralNetworks`.
            classifier:                      is-a `NeuralNetworks` or `ConvolutionalNeuralNetworks`.
            pretext_task_model:             is-a `NeuralNetworks` or `ConvolutionalNeuralNetworks`.

            ssda_loss:                      is-a `SSDAloss`.
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
        if isinstance(encoder, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `encoder` must be `ConvolutionalNeuralNetworks`.")

        if isinstance(classifier, NeuralNetworks) is False and isinstance(classifier, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `classifier` must be `NeuralNetworks` or `ConvolutionalNeuralNetworks`.")
        
        if isinstance(pretext_task_model, NeuralNetworks) is False and isinstance(pretext_task_model, ConvolutionalNeuralNetworks) is False:
            raise TypeError("The type of `pretext_task_model` must be `NeuralNetworks` or `ConvolutionalNeuralNetworks`.")

        if isinstance(ssda_loss, SSDALoss) is False:
            raise TypeError("The type of `ssda_loss` must be `SSDALoss`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        super(SSDAController, self).__init__(**kwargs)

        with self.name_scope():
            self.encoder = encoder
            self.classifier = classifier
            self.pretext_task_model = pretext_task_model

            self.register_child(self.encoder)
            self.register_child(self.classifier)
            self.register_child(self.pretext_task_model)

        self.ssda_loss = ssda_loss

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

            if hybridize_flag is True:
                self.encoder.hybridize()
                self.classifier.hybridize()
                self.pretext_task_model.hybridize()

        except InitDeferredError:
            self.__logger.debug("The initialization should be deferred.")

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__ctx = ctx

        self.__loss_list = []
        self.__acc_list = []
        self.__target_domain_arr = None

        self.__tol = tol
        self.__est = est

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.encoder.collect_params(select)
        params_dict.update(self.classifier.collect_params(select))
        params_dict.update(self.pretext_task_model.collect_params(select))

        return params_dict

    def learn(self, iteratable_data):
        '''
        Learn the observed data points with domain adaptation.

        Args:
            iteratable_data:     is-a `SSDAIterator`.

        '''
        if isinstance(iteratable_data, SSDAIterator) is False:
            raise TypeError("The type of `iteratable_data` must be `SSDAIterator`.")

        self.__loss_list = []
        self.__acc_list = []
        learning_rate = self.__learning_rate
        self.__previous_loss = None
        tol_flag = False
        est_flag = False
        try:
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr, pretext_arr, pretext_label_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                self.batch_size = batch_observed_arr.shape[0]
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    source_encoded_arr = self.encoder.inference(batch_observed_arr)
                    target_encoded_arr = self.encoder.inference(pretext_arr)
                    pred_arr = self.classifier.inference(source_encoded_arr)
                    pretext_pred_arr = self.pretext_task_model.inference(target_encoded_arr)

                    loss, classification_loss, pretext_loss = self.compute_loss(
                        pretext_pred_arr, 
                        pred_arr, 
                        pretext_label_arr, 
                        batch_target_arr, 
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

                    loss.backward()
                    self.trainer.step(batch_observed_arr.shape[0])
                    self.regularize()

                    self.__previous_loss = loss_mean

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        source_encoded_arr = self.encoder.inference(test_batch_observed_arr)
                        target_encoded_arr = self.encoder.inference(pretext_arr)
                        test_pred_arr = self.classifier.inference(source_encoded_arr)
                        pretext_pred_arr = self.pretext_task_model.inference(target_encoded_arr)

                        test_loss, test_classification_loss, test_pretext_loss = self.compute_loss(
                            pretext_pred_arr, 
                            pred_arr, 
                            pretext_label_arr, 
                            batch_target_arr, 
                        )

                        self.__logger.debug("Epochs: " + str(epoch + 1) + " Train total loss: " + str(loss.asnumpy().mean()) + " Test total loss: " + str(test_loss.asnumpy().mean()))
                        self.__logger.debug("Train classification loss: " + str(classification_loss.asnumpy().mean()) + " Test classification loss: " + str(test_classification_loss.asnumpy().mean()))
                        self.__logger.debug("Pretext task loss: " + str(test_pretext_loss.asnumpy().mean()))

                    if self.compute_acc_flag is True:
                        if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                            acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, batch_target_arr)
                            test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_batch_target_arr)
                            if (epoch + 1) % 100 == 0 or epoch < 100:
                                acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, batch_target_arr)
                                test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_batch_target_arr)

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

                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        self.__loss_list.append(
                            (
                                loss.asnumpy().mean(), 
                                test_loss.asnumpy().mean(),
                                classification_loss.asnumpy().mean(),
                                test_classification_loss.asnumpy().mean(),
                                pretext_loss.asnumpy().mean(),
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

                    if tol_flag is True:
                        self.trainer.set_learning_rate(learning_rate)

                else:
                    self.__logger.debug("Early stopping.")
                    self.__logger.debug(str(self.__previous_loss) + " -> " + str(loss_mean))
                    break

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def inference(self, observed_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr)

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
            - `mxnet.ndarray` or `mxnet.symbol` of inferenced label.
        '''
        source_encoded_arr = self.encoder.forward_propagation(F, x)
        pred_arr = self.classifier.forward_propagation(F, source_encoded_arr)

        return pred_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.encoder.regularize()
        self.classifier.regularize()
        self.pretext_task_model.regularize()

    def compute_loss(
        self, 
        pretext_pred_arr, 
        pred_arr, 
        pretext_label_arr, 
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
        return self.ssda_loss(
            pretext_pred_arr, 
            pred_arr, 
            pretext_label_arr, 
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
        inferenced_label_arr = prob_arr.argmax(axis=1)
        answer_label_arr = batch_target_arr.argmax(axis=1)
        acc = (inferenced_label_arr == answer_label_arr).sum() / self.batch_size
        return acc.asnumpy()[0], inferenced_label_arr, answer_label_arr

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
