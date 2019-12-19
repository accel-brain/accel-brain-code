# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t
from pydbm.semisupervised.interface.extractable_centroids import ExtractableCentroids
from pydbm.semisupervised.interface.auto_encodable import AutoEncodable
from pydbm.semisupervised.interface.computable_clustering_loss import ComputableClusteringLoss
from pydbm.semisupervised.computableclusteringloss.balanced_assignments_loss import BalancedAssignmentsLoss
from pydbm.semisupervised.computableclusteringloss.k_means_loss import KMeansLoss
from pydbm.semisupervised.computableclusteringloss.reconstruction_loss import ReconstructionLoss
from pydbm.semisupervised.computableclusteringloss.repelling_loss import RepellingLoss
from pydbm.optimization.optparams.sgd import SGD
from pydbm.params_initializer import ParamsInitializer
from pydbm.nn.nn_layer import NNLayer
from pydbm.synapse.nn_graph import NNGraph
from pydbm.activation.identity_function import IdentityFunction
from pydbm.activation.softmax_function import SoftmaxFunction
from pydbm.activation.softmaxfunction.log_softmax_function import LogSoftmaxFunction

from pydbm.loss.cross_entropy import CrossEntropy
from pydbm.loss.kl_divergence import KLDivergence
from pydbm.verification.verificate_softmax import VerificateSoftmax
from pydbm.optimization.optparams.adam import Adam


class PrototypicalNetworks(object):
    '''
    Prototypical Networks with the Deep Reconstruction-Classification Networks.

    References:
        - Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems (pp. 4077-4087).
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
        nn=None,
        distance_function=None
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

        if distance_function is None:
            def distance_function(arr1, arr2):
                return np.square(arr1 - arr2)

        self.__distance_function = distance_function
        self.__softmax_function = SoftmaxFunction()
        self.__cross_entropy = CrossEntropy()

        self.__learned_observed_arr = None
        self.__learned_target_arr = None
        self.__learned_end_flag = False

    def learn(
        self, 
        np.ndarray observed_arr, 
        np.ndarray target_arr
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of support examples.
            target_arr:     `np.ndarray` of label.
        '''
        self.__learned_end_flag = False
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
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size*2)
                batch_observed_arr = train_observed_arr[rand_index]
                if target_arr is not None:
                    batch_target_arr = train_target_arr[rand_index]
                else:
                    batch_target_arr = None

                try:
                    inferenced_arr = self.inference(batch_observed_arr)
                    clf_loss = self.compute_loss(
                        inferenced_arr,
                        batch_target_arr,
                    )

                    self.back_propagation()
                    self.optimize(learning_rate, epoch)

                    self.__reconstruct(batch_observed_arr)
                    rec_loss = self.__compute_rec_loss()
                    self.__rec_back_propagate()
                    self.__optimize_rec(learning_rate, epoch)

                    loss = (self.__loss_lambda * clf_loss) + ((1 - self.__loss_lambda) * rec_loss)

                    if np.isnan(loss):
                        self.__logger.debug("Epoch: " + str(epoch) + " gradiation may be vanishing. This epoch is skipped.")
                        continue

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
                rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size*2)
                test_batch_observed_arr = test_observed_arr[rand_index]
                if target_arr is not None:
                    test_batch_target_arr = test_target_arr[rand_index]
                else:
                    test_batch_target_arr = None

                test_inferenced_arr = self.inference(
                    test_batch_observed_arr
                )
                test_clf_loss = self.compute_loss(
                    test_inferenced_arr,
                    test_batch_target_arr
                )
                self.__reconstruct(test_batch_observed_arr)
                test_rec_loss = self.__compute_rec_loss()
                self.__rec_back_propagate()
                self.__optimize_rec(learning_rate, epoch)

                test_loss = (self.__loss_lambda * test_clf_loss) + ((1 - self.__loss_lambda) * test_rec_loss)

                if np.isnan(test_loss):
                    self.__logger.debug("Epoch: " + str(epoch) + " gradiation may be vanishing. This epoch is skipped.")
                    continue

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

        self.__learned_observed_arr = observed_arr
        self.__learned_target_arr = target_arr
        self.__learned_end_flag = True

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
        cdef np.ndarray centroid_arr
        cdef np.ndarray distance_arr
        if self.__learned_end_flag is True:
            index_arr = np.arange(self.__learned_observed_arr.shape[0])
            key_arr = np.random.randint(low=0, high=index_arr.shape[0], size=self.__batch_size)
            inferenced_arr = self.forward_propagation(self.__learned_observed_arr[key_arr])
            inferenced_arr = inferenced_arr.reshape((self.__batch_size, -1))
            target_arr = self.__learned_target_arr[key_arr]
            centroid_arr = np.zeros((
                self.__label_n,
                inferenced_arr.shape[1]
            ))
            for i in range(self.__label_n):
                if target_arr[target_arr.argmax(axis=1) == i].shape[0] > 0:
                    centroid_arr[i] = np.nanmean(
                        inferenced_arr[target_arr.argmax(axis=1) == i], 
                        axis=0
                    )

            distance_arr = np.zeros((
                self.__label_n, 
                self.__batch_size,
                inferenced_arr.shape[1],
            ))
            for i in range(self.__label_n):
                distance_arr[i] = self.__distance_function(
                    centroid_arr[i], 
                    pred_arr
                )

            distance_arr = distance_arr.transpose((1, 0, 2))
            distance_arr = np.nansum(distance_arr, axis=2)
            distance_arr = distance_arr / distance_arr.shape[-1]
            distance_arr = self.__softmax_function.activate(-distance_arr)

            return distance_arr
        else:
            return pred_arr

    def forward_propagation(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points and do soft assignment.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of result of soft assignment.
        '''
        cdef np.ndarray feature_arr
        cdef np.ndarray support_arr
        cdef np.ndarray query_arr

        if observed_arr.shape[0] == self.__batch_size:
            feature_arr = self.__auto_encodable.embed_feature_points(observed_arr)
        else:
            support_arr = observed_arr[:self.__batch_size]
            query_arr = observed_arr[self.__batch_size:]
            feature_arr = np.r_[
                self.__auto_encodable.embed_feature_points(support_arr),
                self.__auto_encodable.embed_feature_points(query_arr)
            ]

        if feature_arr.ndim != 2:
            default_shape = feature_arr.copy().shape
            if feature_arr.shape[0] != self.__batch_size:
                default_shape_list = list(default_shape)
                default_shape_list[0] = self.__batch_size
                default_shape = tuple(default_shape_list)
            feature_arr = feature_arr.reshape((feature_arr.shape[0], -1))
        else:
            default_shape = None

        self.__default_shape = default_shape

        return feature_arr

    def compute_loss(self, inferenced_arr, target_arr):
        '''
        Compute loss.

        Args:
            inferenced_arr:      `np.ndarray` of result of softmax output layer.
            target_arr:          `np.ndarray` of label data.
        
        Returns:
            (loss, `np.ndarray` of delta)
        '''
        self.__inferenced_arr = inferenced_arr
        self.__target_arr = target_arr

        support_inferenced_arr = inferenced_arr[:self.__batch_size]
        query_inferenced_arr = inferenced_arr[self.__batch_size:]
        support_target_arr = target_arr[:self.__batch_size]
        query_target_arr = target_arr[self.__batch_size:]

        cdef np.ndarray centroid_arr = np.zeros((
            self.__label_n,
            support_inferenced_arr.shape[1]
        ))
        for i in range(self.__label_n):
            if support_target_arr[support_target_arr.argmax(axis=1) == i].shape[0] > 0:
                centroid_arr[i] = np.nanmean(
                    support_inferenced_arr[support_target_arr.argmax(axis=1) == i], 
                    axis=0
                )

        distance_arr = np.zeros((
            self.__label_n, 
            self.__batch_size,
            support_inferenced_arr.shape[1],
        ))
        for i in range(self.__label_n):
            distance_arr[i] = self.__distance_function(
                centroid_arr[i], 
                query_inferenced_arr
            )

        distance_arr = distance_arr.transpose((1, 0, 2))
        #self.__distance_arr = distance_arr
        distance_arr = np.nansum(distance_arr, axis=2)
        distance_arr = distance_arr / distance_arr.shape[-1]

        distance_arr = self.__softmax_function.activate(-distance_arr)

        print(
            (distance_arr.argmax(axis=1) == query_target_arr.argmax(axis=1)).astype(int).sum() / self.__batch_size
        )

        distance = -np.nansum(
            np.ma.log(
                distance_arr[np.arange(self.__batch_size), query_target_arr.argmax(axis=1)]
            )
        ) / self.__batch_size

        self.__distance = distance
        self.__distance_arr = distance_arr
        self.__target_arr = target_arr
        self.__centroid_arr = centroid_arr
        self.__support_inferenced_arr = support_inferenced_arr
        self.__query_inferenced_arr = query_inferenced_arr
        self.__query_target_arr = query_target_arr

        return distance

    def __reconstruct(self, observed_arr):
        cdef np.ndarray reconstructed_arr = self.__auto_encodable.inference(observed_arr[:self.__batch_size])
        self.__reconstructed_arr = reconstructed_arr
        self.__observed_arr = observed_arr[:self.__batch_size]

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
        cdef np.ndarray delta_arr = self.__distance_arr
        delta_arr[np.arange(delta_arr.shape[0]), self.__query_target_arr.argmax(axis=1)] -= 1

        delta_arr = np.repeat(
            np.expand_dims(delta_arr, axis=2), 
            repeats=self.__query_inferenced_arr.shape[-1], 
            axis=2
        ) / self.__query_inferenced_arr.shape[-1]

        """
        for i in range(self.__label_n):
            row = delta_arr[self.__query_target_arr.argmax(axis=1) == i].shape[0]
            if row > 0:
                delta_arr[self.__query_target_arr.argmax(axis=1) == i] += self.__centroid_arr[i] / row
        """
        delta_arr = np.nansum(delta_arr, axis=1)
        delta_arr = delta_arr * self.__loss_lambda

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
