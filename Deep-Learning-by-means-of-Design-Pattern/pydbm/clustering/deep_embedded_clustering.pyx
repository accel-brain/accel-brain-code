# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t
from pydbm.loss.kl_divergence import KLDivergence
from pydbm.clustering.interface.extractable_centroids import ExtractableCentroids
from pydbm.clustering.interface.auto_encodable import AutoEncodable
from pydbm.optimization.optparams.sgd import SGD


class DeepEmbeddedClustering(object):
    '''
    The Deep Embedded Clustering(DEC).

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    # Degrees of freedom of the Student's t-distribution.
    __alpha = 1

    def get_alpha(self):
        ''' getter or degrees of freedom of the Student's t-distribution. '''
        return self.__alpha
    
    def set_alpha(self, value):
        ''' setter for degrees of freedom of the Student's t-distribution. '''
        self.__alpha = value

    alpha = property(get_alpha, set_alpha)

    # KL Divergence.
    __kl_divergence = KLDivergence()

    def __init__(
        self,
        auto_encodable,
        extractable_centroids,
        int k=10,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        opt_params=None,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        grad_clip_threshold=1.0
    ):
        '''
        Init.

        Args:
            auto_encodable:                 is-a `AutoEncodable`.
            extractable_centroids:          is-a `ExtractableCentroids`.
            k:                              The number of clusters.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            opt_params:                     is-a `OptParams`. If `None`, this value will be `SGD`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                            Tolerance for the optimization.
            tld:                            Tolerance for deviation of loss.
            grad_clip_threshold:            Threshold of the gradient clipping.
        '''
        if isinstance(auto_encodable, AutoEncodable) is False:
            raise TypeError("The type of `auto_encodable` must be `AutoEncodable`.")

        if isinstance(extractable_centroids, ExtractableCentroids) is False:
            raise TypeError("The type `extractable_centroids` must be `ExtractableCentroids`.")

        self.__auto_encodable = auto_encodable
        self.__extractable_centroids = extractable_centroids
        self.__k = k
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        if opt_params is None:
            self.__opt_params = SGD()
            self.__opt_params.weight_limit = 1e+10
            self.__opt_params.dropout_rate = 0.2
        else:
            self.__opt_params = opt_params
        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld
        self.__grad_clip_threshold = grad_clip_threshold
        logger = getLogger("pydbm")
        self.__logger = logger

    def __setup_initial_centroids(self, np.ndarray observed_arr):
        '''
        Create initial centroids and set it as property.

        Args:
            observed_arr:               `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of centroids.
        '''
        cdef np.ndarray key_arr
        if observed_arr.shape[0] != self.__batch_size:
            key_arr = np.arange(observed_arr.shape[0])
            np.random.shuffle(key_arr)
            observed_arr = observed_arr[key_arr[:self.__batch_size]]

        feature_arr = self.__auto_encodable.embed_feature_points(observed_arr)
        self.__mu_arr = self.__extractable_centroids.extract_centroids(feature_arr, self.__k)

    def learn(self, np.ndarray observed_arr):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
        '''
        self.__auto_encodable.pre_learn(observed_arr)
        self.__setup_initial_centroids(observed_arr)

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray train_observed_arr
        cdef np.ndarray test_observed_arr

        cdef np.ndarray rand_index
        cdef np.ndarray batch_observed_arr

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round(self.__test_size_rate * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
        else:
            train_observed_arr = observed_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray pred_arr
        cdef np.ndarray test_pred_arr

        loss_log_list = []
        try:
            loss_list = []
            for epoch in range(self.__epochs):
                self.__auto_encodable.inferencing_mode = False
                self.__opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    loss = self.compute_loss(
                        pred_arr,
                        self.compute_target_distribution(pred_arr)
                    )
                    self.back_propagation()
                    self.optimize(learning_rate, epoch)

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        break
                    else:
                        raise

                test_loss = 0.0
                if self.__test_size_rate > 0:
                    self.__opt_params.inferencing_mode = True
                    self.__auto_encodable.inferencing_mode = True

                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]

                    test_pred_arr = self.inference(
                        test_batch_observed_arr
                    )
                    test_loss = self.compute_loss(
                        test_pred_arr,
                        self.compute_target_distribution(test_pred_arr)
                    )

                loss_list.append(loss)
                loss_log_list.append((loss, test_loss))
                self.__logger.debug("Epoch: " + str(epoch) + " Train loss(KLD): " + str(loss) + " Test loss(KLD): " + str(test_loss))

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")
        self.__loss_arr = np.array(loss_log_list)

    def clustering(self, np.ndarray observed_arr):
        '''
        Clustering.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            `np.ndarray` of labels.
        '''
        cdef np.ndarray q_arr = self.forward_propagation(observed_arr)
        if q_arr.shape[2] > 1:
            q_arr = np.nanmean(q_arr, axis=2)
        q_arr = q_arr.reshape((q_arr.shape[0], q_arr.shape[1]))
        return q_arr.argmax(axis=1)

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

        cdef int k = self.__mu_arr.shape[0]
        cdef int dim = feature_arr.shape[1]
        cdef np.ndarray[DOUBLE_t, ndim=3] q_arr = np.zeros((
            batch_size, 
            k, 
            dim
        ))
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_arr = np.zeros((
            batch_size, 
            k, 
            dim
        ))
        for i in range(k):
            delta_arr[:, i] = feature_arr - self.__mu_arr[i]
            q_arr[:, i] = np.power((1 + np.square(delta_arr[:, i]) / self.__alpha), -(self.__alpha + 1) / 2)
        q_arr = q_arr / np.nansum(q_arr, axis=1).reshape((batch_size, 1, q_arr.shape[2]))

        self.__feature_arr = feature_arr
        self.__delta_arr = delta_arr
        self.__default_shape = default_shape

        return q_arr

    def compute_target_distribution(self, q_arr):
        '''
        Compute target distribution.

        Args:
            q_arr:            `np.ndarray` of result of soft assignment.

        Returns:
            `np.ndarray` of target distribution.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] f_arr = np.nansum(q_arr, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=3] p_arr = np.power(q_arr, 2) / f_arr.reshape((f_arr.shape[0], f_arr.shape[1], 1))
        p_arr = p_arr / np.nansum(p_arr, axis=1).reshape((p_arr.shape[0], 1, p_arr.shape[2]))
        return p_arr
    
    def compute_loss(self, p_arr, q_arr):
        '''
        Compute loss.

        Args:
            p_arr:      `np.ndarray` of result of soft assignment.
            q_arr:      `np.ndarray` of target distribution.
        
        Returns:
            (loss, `np.ndarray` of delta)
        '''
        loss = self.__kl_divergence.compute_loss(p_arr, q_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_z_arr = ((self.__alpha + 1) / self.__alpha) * np.nansum(
            np.power(1 + np.square(self.__delta_arr) / self.__alpha, -1) * (p_arr - q_arr) * np.square(self.__delta_arr), 
            axis=1
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_mu_arr = -((self.__alpha + 1) / self.__alpha) * np.nansum(
            np.power(1 + np.square(self.__delta_arr) / self.__alpha, -1)  * (p_arr - q_arr) * np.square(self.__delta_arr),
            axis=2
        )

        self.__delta_z_arr = delta_z_arr
        self.__delta_mu_arr = np.dot(self.__feature_arr.T, delta_mu_arr).T
        self.__delta_z_arr = self.__grad_clipping(self.__delta_z_arr)
        self.__delta_mu_arr = self.__grad_clipping(self.__delta_mu_arr)

        return loss

    def __grad_clipping(self, diff_arr):
        v = np.linalg.norm(diff_arr)
        if v > self.__grad_clip_threshold:
            diff_arr = diff_arr * self.__grad_clip_threshold / v

        return diff_arr

    def back_propagation(self):
        '''
        Back propagation.

        Returns:
            `np.ndarray` of delta.
        '''
        self.__auto_encodable.backward_auto_encoder(self.__delta_z_arr.reshape(self.__default_shape))

    def optimize(self, learning_rate, epoch):
        '''
        Optimize.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        params_list = [
            self.__mu_arr
        ]
        grads_list = [
            self.__delta_mu_arr
        ]
        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.__mu_arr = params_list[0]
        self.__auto_encodable.optimize_auto_encoder(learning_rate, epoch)

    def get_mu_arr(self):
        ''' getter for learned centroids. '''
        return self.__mu_arr
    
    def set_mu_arr(self, value):
        ''' setter for learned centroids. '''
        self.__mu_arr = value
    
    mu_arr = property(get_mu_arr, set_mu_arr)

    def get_loss_arr(self):
        ''' getter '''
        return self.__loss_arr
    
    def set_loss_arr(self, value):
        ''' setter '''
        self.__loss_arr = value
    
    loss_arr = property(get_loss_arr, set_loss_arr)

    def get_auto_encodable(self):
        ''' getter '''
        return self.__auto_encodable
    
    def set_auto_encodable(self, value):
        ''' setter '''
        self.__auto_encodable = value
    
    auto_encodable = property(get_auto_encodable, set_auto_encodable)
