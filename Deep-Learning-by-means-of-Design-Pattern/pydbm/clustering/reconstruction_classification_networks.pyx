# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t
from pydbm.clustering.interface.extractable_centroids import ExtractableCentroids
from pydbm.clustering.interface.auto_encodable import AutoEncodable
from pydbm.clustering.interface.computable_clustering_loss import ComputableClusteringLoss
from pydbm.clustering.computableclusteringloss.balanced_assignments_loss import BalancedAssignmentsLoss
from pydbm.clustering.computableclusteringloss.k_means_loss import KMeansLoss
from pydbm.clustering.computableclusteringloss.reconstruction_loss import ReconstructionLoss
from pydbm.clustering.computableclusteringloss.repelling_loss import RepellingLoss
from pydbm.optimization.optparams.sgd import SGD
from pydbm.params_initializer import ParamsInitializer
from pydbm.nn.neural_network import NeuralNetwork
from pydbm.nn.nn_layer import NNLayer
from pydbm.synapse.nn_graph import NNGraph
from pydbm.activation.softmax_function import SoftmaxFunction
from pydbm.loss.cross_entropy import CrossEntropy
from pydbm.loss.kl_divergence import KLDivergence
from pydbm.verification.verificate_softmax import VerificateSoftmax
from pydbm.optimization.optparams.adam import Adam


class ReconstructionClassificationNetworks(object):
    '''
    The Deep Reconstruction-Classification Networks.

    References:
        - Ghifary, M., Kleijn, W. B., Zhang, M., Balduzzi, D., & Li, W. (2016, October). Deep reconstruction-classification networks for unsupervised domain adaptation. In European Conference on Computer Vision (pp. 597-613). Springer, Cham.
    '''

    def __init__(
        self,
        auto_encodable,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double loss_lambda=0.7,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        denoising_flag=True,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0},
        nn=None
    ):
        '''
        Init.

        Args:
            auto_encodable:                     is-a `AutoEncodable`.
            epochs:                             Epochs of Mini-batch.
            bath_size:                          Batch size of Mini-batch.
            learning_rate:                      Learning rate.
            learning_attenuate_rate:            Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                    Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                                Additionally, in relation to regularization,
                                                this class constrains weight matrixes every `attenuate_epoch`.

            loss_lambda:                        Trade-off parameter of loss functions.

            test_size_rate:                     Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                                Tolerance for the optimization.
            tld:                                Tolerance for deviation of loss.
            denoising_flag:                     Do pre-learning as a Denoising Auto-Encoder or not.
            params_initializer:                 is-a `ParamsInitializer`.
                                                If `denoising_flag` is `True`, this class will noise 
                                                observed data points by using this `params_initializer`.

            params_dict:                        `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
            nn:                                 `NeuralNetwork` of output layer of classifier.
        '''
        if isinstance(auto_encodable, AutoEncodable) is False:
            raise TypeError("The type of `auto_encodable` must be `AutoEncodable`.")

        self.__auto_encodable = auto_encodable

        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__loss_lambda = loss_lambda

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__denoising_flag = denoising_flag
        self.__params_initializer = params_initializer
        self.__params_dict = params_dict

        logger = getLogger("pydbm")
        self.__logger = logger

        if isinstance(nn, NeuralNetwork) is True or nn is None:
            self.__nn = nn
        else:
            raise TypeError()

    def learn(
        self, 
        np.ndarray observed_arr, 
        np.ndarray target_arr,
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of noised observed data points.
        '''
        self.__label_n = target_arr.shape[1]

        cdef np.ndarray noised_observed_arr
        if self.__denoising_flag is True:
            noised_observed_arr = observed_arr + self.__params_initializer.sample(
                size=observed_arr.copy().shape,
                **self.__params_dict
            )
        else:
            noised_observed_arr = observed_arr

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray train_observed_arr
        cdef np.ndarray test_observed_arr

        cdef np.ndarray rand_index
        cdef np.ndarray batch_observed_arr

        cdef np.ndarray train_target_arr
        cdef np.ndarray test_target_arr

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round(self.__test_size_rate * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
            if target_arr is not None:
                train_target_arr = target_arr[train_index]
                test_target_arr = target_arr[test_index]
        else:
            train_observed_arr = observed_arr
            if target_arr is not None:
                train_target_arr = target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray q_arr
        cdef np.ndarray test_q_arr
        cdef np.ndarray p_arr
        cdef np.ndarray test_p_arr

        loss_log_list = []
        clf_loss_log_list = []
        rec_loss_log_list = []
        try:
            loss_list = []
            for epoch in range(self.__epochs):
                self.__auto_encodable.inferencing_mode = False
                if self.__nn is not None:
                    self.__nn.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                if target_arr is not None:
                    batch_target_arr = train_target_arr[rand_index]
                else:
                    batch_target_arr = None

                try:
                    inferenced_arr = self.inference(batch_observed_arr)
                    clf_loss = self.compute_loss(
                        inferenced_arr,
                        batch_target_arr
                    )
                    self.back_propagation()
                    self.optimize(learning_rate, epoch)

                    self.__reconstruct(batch_observed_arr)
                    rec_loss = self.__compute_rec_loss()
                    self.__rec_back_propagate()
                    self.__optimize_rec(learning_rate, epoch)

                    loss = (self.__loss_lambda * clf_loss) + ((1 - self.__loss_lambda) * rec_loss)

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        break
                    else:
                        raise

                test_loss = 0.0
                self.__auto_encodable.inferencing_mode = True
                if self.__nn is not None:
                    self.__nn.opt_params.inferencing_mode = True

                rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                test_batch_observed_arr = test_observed_arr[rand_index]
                if target_arr is not None:
                    test_batch_target_arr = test_target_arr[rand_index]
                else:
                    test_batch_target_arr = None

                test_inferneced_arr = self.inference(
                    test_batch_observed_arr
                )
                test_clf_loss = self.compute_loss(
                    test_inferneced_arr,
                    test_batch_target_arr
                )
                self.__reconstruct(test_batch_observed_arr)
                test_rec_loss = self.__compute_rec_loss()
                self.__rec_back_propagate()
                self.__optimize_rec(learning_rate, epoch)

                test_loss = (self.__loss_lambda * test_clf_loss) + ((1 - self.__loss_lambda) * test_rec_loss)

                loss_list.append(loss)
                loss_log_list.append((loss, test_loss))
                clf_loss_log_list.append((clf_loss, test_clf_loss))
                rec_loss_log_list.append((rec_loss, test_rec_loss))
                self.__logger.debug("Epoch: " + str(epoch) + " Train loss: " + str(loss) + " Test loss: " + str(test_loss))
                self.__logger.debug("Train classification loss: " + str(clf_loss) + " Test classification loss: " + str(test_clf_loss))
                self.__logger.debug("Train reconstruction loss: " + str(rec_loss) + " Test reconstruction loss: " + str(test_rec_loss))

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")
        self.__loss_arr = np.array(loss_log_list)
        self.__clf_loss_arr = np.array(clf_loss_log_list)
        self.__rec_loss_arr = np.array(rec_loss_log_list)

    def inference(self, np.ndarray observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        cdef np.ndarray pred_arr = self.forward_propagation(
            observed_arr
        )
        return pred_arr

    def forward_propagation(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points and do soft assignment.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of result of soft assignment.
        '''
        cdef np.ndarray feature_arr = self.__auto_encodable.embed_feature_points(observed_arr)

        cdef int batch_size = feature_arr.shape[0]
        if feature_arr.ndim != 2:
            default_shape = feature_arr.copy().shape
            feature_arr = feature_arr.reshape((batch_size, -1))
        else:
            default_shape = None

        if self.__nn is None:
            nn_layer = NNLayer(
                NNGraph(
                    activation_function=SoftmaxFunction(),
                    hidden_neuron_count=feature_arr.shape[1],
                    output_neuron_count=self.__label_n,
                    scale=1.0,
                    params_initializer=ParamsInitializer(),
                    params_dict={"loc": 0.0, "scale": 1.0}
                )
            )
            self.__nn = NeuralNetwork(
                nn_layer_list=[nn_layer],
                computable_loss=CrossEntropy(),
                opt_params=Adam(),
                verificatable_result=VerificateSoftmax(),
            )
            self.__nn.opt_params.inferencing_mode = True

        cdef np.ndarray inferenced_arr = self.__nn.inference(feature_arr)
        self.__default_shape = default_shape

        return inferenced_arr

    def compute_loss(self, inferenced_arr, target_arr):
        '''
        Compute loss.

        Args:
            inferneced_arr:      `np.ndarray` of result of softmax output layer.
            target_arr:          `np.ndarray` of labeled data.
        
        Returns:
            (loss, `np.ndarray` of delta)
        '''
        self.__inferenced_arr = inferenced_arr
        self.__target_arr = target_arr
        clf_loss = self.__nn.computable_loss.compute_loss(inferenced_arr, target_arr)
        return clf_loss

    def __reconstruct(self, observed_arr):
        cdef np.ndarray reconstructed_arr = self.__auto_encodable.inference(observed_arr)
        self.__reconstructed_arr = reconstructed_arr
        self.__observed_arr = observed_arr

    def __compute_rec_loss(self):
        rec_loss = self.__auto_encodable.auto_encoder_model.computable_loss.compute_loss(
            self.__reconstructed_arr,
            self.__observed_arr
        )
        return rec_loss

    def back_propagation(self):
        '''
        Back propagation.

        Returns:
            `np.ndarray` of delta.
        '''
        cdef np.ndarray delta_arr = self.__nn.computable_loss.compute_delta(
            self.__inferenced_arr,
            self.__target_arr
        )
        delta_arr = delta_arr * self.__loss_lambda
        delta_arr = self.__nn.back_propagation(delta_arr)
        if self.__default_shape is not None:
            delta_arr = delta_arr.reshape(self.__default_shape)

        self.__auto_encodable.backward_auto_encoder(
            delta_arr,
            encoder_only_flag=True
        )

    def __rec_back_propagate(self):
        cdef np.ndarray rec_delta_arr = self.__auto_encodable.auto_encoder_model.computable_loss.compute_delta(
            self.__reconstructed_arr,
            self.__observed_arr
        )
        rec_delta_arr = rec_delta_arr * (1 - self.__loss_lambda)
        self.__auto_encodable.backward_auto_encoder(
            rec_delta_arr,
            encoder_only_flag=False
        )

    def optimize(self, learning_rate, epoch):
        '''
        Optimize.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        self.__nn.optimize(learning_rate, epoch)
        self.__auto_encodable.optimize_auto_encoder(
            learning_rate, 
            epoch, 
            encoder_only_flag=True
        )

    def __optimize_rec(self, learning_rate, epoch):
        self.__auto_encodable.optimize_auto_encoder(
            learning_rate, 
            epoch, 
            encoder_only_flag=False
        )

    def get_auto_encodable(self):
        ''' getter '''
        return self.__auto_encodable
    
    def set_auto_encodable(self, value):
        ''' setter '''
        self.__auto_encodable = value
    
    auto_encodable = property(get_auto_encodable, set_auto_encodable)

    def get_nn(self):
        ''' getter '''
        return self.__nn
    
    def set_nn(self, value):
        ''' setter '''
        if isinstance(value, NeuralNetwork) is True:
            self.__nn = value
        else:
            raise TypeError()
    
    nn = property(get_nn, set_nn)

    def get_loss_arr(self):
        ''' getter '''
        return self.__loss_arr
    
    def set_loss_arr(self, value):
        ''' setter '''
        self.__loss_arr = value
    
    loss_arr = property(get_loss_arr, set_loss_arr)

    def get_clf_loss_arr(self):
        ''' getter '''
        return self.__clf_loss_arr
    
    def set_clf_loss_arr(self, value):
        ''' setter '''
        self.__clf_loss_arr = value
    
    clf_loss_arr = property(get_clf_loss_arr, set_clf_loss_arr)

    def get_rec_loss_arr(self):
        ''' getter '''
        return self.__rec_loss_arr
    
    def set_rec_loss_arr(self, value):
        ''' setter '''
        self.__rec_loss_arr = value
    
    rec_loss_arr = property(get_rec_loss_arr, set_rec_loss_arr)
