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
from pydbm.loss.mean_squared_error import MeanSquaredError


class DeepEmbeddedClustering(object):
    '''
    The Deep Embedded Clustering(DEC).

    References:
        - Aljalbout, E., Golkov, V., Siddiqui, Y., Strobel, M., & Cremers, D. (2018). Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    # KL Divergence.
    __kl_divergence = KLDivergence()
    # MSE
    __mean_squared_error = MeanSquaredError()

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
        grad_clip_threshold=1.0,
        T=100,
        alpha=1,
        soft_assign_weight=0.1,
        beta=0.33,
        gamma=0.33,
        kappa=0.33,
        repelling_weight=1.0,
        anti_repelling_weight=1.0
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
            T:                              Target distribution update interval.
            alpha:                          Degrees of freedom of the Student's t-distribution.
            beta:                           Weight of balanced assignments loss.
            gamma:                          A coefficient that controls the degree of distorting embedded space.
            kappa:                          Weight of K-Means loss.
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

        self.__T = T

        self.__alpha = alpha

        self.__soft_assign_weight = soft_assign_weight
        self.__beta = beta
        self.__gamma = gamma
        self.__kappa = kappa

        self.__repelling_weight = repelling_weight
        self.__anti_repelling_weight = anti_repelling_weight

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
        cdef np.ndarray q_arr
        cdef np.ndarray test_q_arr
        cdef np.ndarray p_arr
        cdef np.ndarray test_p_arr

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
                    q_arr = self.inference(batch_observed_arr)

                    if epoch == 0 or epoch % self.__T == 0:
                        p_arr = self.compute_target_distribution(q_arr)

                    q_arr = q_arr.copy()

                    loss = self.compute_loss(
                        q_arr,
                        p_arr
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

                    test_q_arr = self.inference(
                        test_batch_observed_arr
                    )

                    if epoch == 0 or epoch % self.__T == 0:
                        test_p_arr = self.compute_target_distribution(test_q_arr)

                    test_loss = self.compute_loss(
                        test_q_arr,
                        test_p_arr
                    )

                loss_list.append(loss)
                loss_log_list.append((loss, test_loss))
                self.__logger.debug("Epoch: " + str(epoch) + " Train loss: " + str(loss) + " Test loss: " + str(test_loss))

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
        return self.__assign_label(q_arr)

    def __assign_label(self, q_arr):
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

        self.__observed_arr = observed_arr
        self.__pred_arr = self.auto_encodable.inference(observed_arr)

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
        self.__delta_z_arr = self.__delta_z_arr * self.__soft_assign_weight
        self.__delta_mu_arr = self.__delta_mu_arr * self.__soft_assign_weight

        # K-Means loss.
        cdef np.ndarray label_arr = self.__assign_label(q_arr)
        cdef np.ndarray t_hot_arr = np.zeros((label_arr.shape[0], self.__delta_arr.shape[1]))
        for i in range(label_arr.shape[0]):
            t_hot_arr[i, label_arr[i]] = 1
        t_hot_arr = np.expand_dims(t_hot_arr, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_kmeans_arr = t_hot_arr.astype(np.float) * np.square(self.__delta_arr)
        self.__delta_kmeans_z_arr = np.nanmean(delta_kmeans_arr, axis=1)
        self.__delta_kmeans_z_arr = self.__grad_clipping(self.__delta_kmeans_z_arr)
        kmeans_loss = np.nansum(self.__delta_kmeans_z_arr)
        self.__delta_kmeans_z_arr = self.__delta_kmeans_z_arr * self.__kappa

        # Balanced assignments loss
        cdef assign_arr = q_arr
        if assign_arr.shape[2] > 1:
            assign_arr = np.nanmean(assign_arr, axis=2)
        assign_arr = assign_arr.reshape((assign_arr.shape[0], assign_arr.shape[1]))

        cdef uniform_arr = np.random.uniform(
            low=assign_arr.min(), 
            high=assign_arr.max(), 
            size=assign_arr.copy().shape
        )
        ba_loss = self.__kl_divergence.compute_loss(
            assign_arr,
            uniform_arr
        )
        cdef np.ndarray delta_ba_arr = self.__kl_divergence.compute_delta(
            assign_arr,
            uniform_arr
        )
        delta_ba_arr = delta_ba_arr * self.__beta
        self.__delta_ba_arr = np.dot(self.__feature_arr.T, delta_ba_arr).T
        self.__delta_ba_arr = self.__grad_clipping(self.__delta_ba_arr)

        # Reconstruction Loss.
        reconstructed_loss = self.__mean_squared_error.compute_loss(self.__pred_arr, self.__observed_arr)
        cdef np.ndarray delta_rec_arr = self.__mean_squared_error.compute_delta(self.__pred_arr, self.__observed_arr)

        self.__delta_rec_arr = self.__grad_clipping(delta_rec_arr)
        self.__delta_rec_arr = self.__delta_rec_arr * self.__gamma

        # Repelling penalty.
        cdef int N = self.__feature_arr.reshape((self.__feature_arr.shape[0], -1)).shape[1]
        cdef int s
        cdef int sa
        cdef np.ndarray pt_arr
        cdef np.ndarray anti_arr
        cdef np.ndarray penalty_arr = np.zeros(label_arr.shape[0])
        cdef np.ndarray anti_penalty_arr = np.zeros(label_arr.shape[0])

        for label in label_arr:
            feature_arr = self.__feature_arr[label_arr == label]
            anti_feature_arr = self.__feature_arr[label_arr != label]
            s = feature_arr.shape[0]
            sa = anti_feature_arr.shape[0]

            pt_arr = np.zeros(s ** 2)
            anti_arr = np.zeros(sa * s)
            k = 0
            l = 0
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    pt_arr[k] = np.dot(feature_arr[i].T, feature_arr[j]) / (np.sqrt(np.dot(feature_arr[i], feature_arr[i])) * np.sqrt(np.dot(feature_arr[j], feature_arr[j])))
                    k += 1
                for j in range(sa):
                    anti_arr[l] = np.dot(feature_arr[i].T, anti_feature_arr[j]) / (np.sqrt(np.dot(feature_arr[i], feature_arr[i])) * np.sqrt(np.dot(anti_feature_arr[j], anti_feature_arr[j])))
                    l += 1

            penalty_arr[label] = np.nansum(pt_arr) / (N * (N - 1))
            anti_penalty_arr[label] = np.nansum(anti_arr) / (N * (N - 1))

        penalty = penalty_arr.mean()
        anti_penalty = anti_penalty_arr.mean()

        # The number of delta.
        penalty = penalty / 4
        penalty = penalty * self.__repelling_weight
        anti_penalty = anti_penalty / 4
        anti_penalty = anti_penalty * self.__anti_repelling_weight
        penalty_term = (penalty + anti_penalty)

        self.__delta_mu_arr = self.__delta_mu_arr + penalty_term
        self.__delta_z_arr = self.__delta_z_arr + penalty_term
        self.__delta_kmeans_z_arr = self.__delta_kmeans_z_arr + penalty_term
        self.__delta_ba_arr = self.__delta_ba_arr + penalty_term
        #self.__delta_rec_arr = self.__delta_rec_arr + penalty_term

        return np.array([
            np.abs(self.__delta_mu_arr).mean(), 
            np.abs(self.__delta_z_arr).mean(), 
            np.abs(self.__delta_kmeans_z_arr).mean(), 
            np.abs(self.__delta_ba_arr).mean(), 
            np.abs(self.__delta_rec_arr).mean(), 
        ]).mean()

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
        self.__auto_encodable.backward_auto_encoder(
            self.__delta_rec_arr,
            encoder_only_flag=False
        )
        cdef np.ndarray delta_arr = self.__delta_z_arr.reshape(
            self.__default_shape
        ) + self.__delta_kmeans_z_arr.reshape(
            self.__default_shape
        )
        self.__auto_encodable.backward_auto_encoder(
            self.__grad_clipping(delta_arr)
        )

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
            self.__grad_clipping(self.__delta_mu_arr + self.__delta_ba_arr)
        ]
        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.__mu_arr = params_list[0]
        self.__auto_encodable.optimize_auto_encoder(learning_rate, epoch, encoder_only_flag=False)

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

    def get_opt_params(self):
        ''' getter '''
        return self.__opt_params
    
    def set_opt_params(self, value):
        ''' setter '''
        self.__opt_params = value
    
    opt_params = property(get_opt_params, set_opt_params)
