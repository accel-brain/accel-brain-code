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


class DeepEmbeddedClustering(object):
    '''
    The Deep Embedded Clustering(DEC).

    References:
        - Aljalbout, E., Golkov, V., Siddiqui, Y., Strobel, M., & Cremers, D. (2018). Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Ren, Y., Hu, K., Dai, X., Pan, L., Hoi, S. C., & Xu, Z. (2019). Semi-supervised deep embedded clustering. Neurocomputing, 325, 121-130.
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
        - Wagstaff, K., Cardie, C., Rogers, S., & Schrödl, S. (2001, June). Constrained k-means clustering with background knowledge. In Icml (Vol. 1, pp. 577-584).
    '''

    def __init__(
        self,
        auto_encodable,
        extractable_centroids,
        computable_clustering_loss_list=[],
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
        soft_assign_weight=0.25,
        pairwise_lambda=0.25,
        denoising_flag=True,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.

        Args:
            auto_encodable:                     is-a `AutoEncodable`.
            extractable_centroids:              is-a `ExtractableCentroids`.
            computable_clustering_loss_list:    `list` of `ComputableClusteringLoss`s.
            k:                                  The number of clusters.
            epochs:                             Epochs of Mini-batch.
            bath_size:                          Batch size of Mini-batch.
            learning_rate:                      Learning rate.
            learning_attenuate_rate:            Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                    Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                                Additionally, in relation to regularization,
                                                this class constrains weight matrixes every `attenuate_epoch`.

            opt_params:                         is-a `OptParams`. If `None`, this value will be `SGD`.

            test_size_rate:                     Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                                Tolerance for the optimization.
            tld:                                Tolerance for deviation of loss.
            grad_clip_threshold:                Threshold of the gradient clipping.
            T:                                  Target distribution update interval.
            alpha:                              Degrees of freedom of the Student's t-distribution.
            soft_assign_weight:                 Weight of soft assignments.
            pairwise_lambda:                    Weight of pairwise constraint.
            denoising_flag:                     Do pre-learning as a Denoising Auto-Encoder or not.
            params_initializer:                 is-a `ParamsInitializer`.
                                                If `denoising_flag` is `True`, this class will noise 
                                                observed data points by using this `params_initializer`.

            params_dict:                        `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(auto_encodable, AutoEncodable) is False:
            raise TypeError("The type of `auto_encodable` must be `AutoEncodable`.")

        if isinstance(extractable_centroids, ExtractableCentroids) is False:
            raise TypeError("The type `extractable_centroids` must be `ExtractableCentroids`.")

        reconstruction_loss_flag = False
        for computable_clustering_loss in computable_clustering_loss_list:
            if isinstance(computable_clustering_loss, ComputableClusteringLoss) is False:
                raise TypeError()
            if isinstance(computable_clustering_loss, ReconstructionLoss) is True:
                reconstruction_loss_flag = True

        if len(computable_clustering_loss_list) == 0:
            computable_clustering_loss_list = [
                BalancedAssignmentsLoss(weight=0.125),
                KMeansLoss(weight=0.125),
                ReconstructionLoss(weight=0.125),
                RepellingLoss(weight=0.125),
            ]
        else:
            if reconstruction_loss_flag is False:
                computable_clustering_loss_list.append(
                    ReconstructionLoss(
                        weight=0.0
                    )
                )

        self.__auto_encodable = auto_encodable
        self.__extractable_centroids = extractable_centroids
        self.__computable_clustering_loss_list = computable_clustering_loss_list

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
        self.__pairwise_lambda = pairwise_lambda

        self.__denoising_flag = denoising_flag
        self.__params_initializer = params_initializer
        self.__params_dict = params_dict

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
        cdef np.ndarray _observed_arr

        if observed_arr.shape[0] != self.__batch_size:
            epochs = observed_arr.shape[0] / self.__batch_size
            epochs = int(epochs)
            feature_arr = None
            for epoch in range(epochs):
                _observed_arr = observed_arr[
                    epoch * self.__batch_size:(epoch + 1) * self.__batch_size
                ]
                _feature_arr = self.__auto_encodable.embed_feature_points(_observed_arr)
                if feature_arr is None:
                    feature_arr = _feature_arr
                else:
                    feature_arr = np.r_[feature_arr, _feature_arr]
        else:
            feature_arr = self.__auto_encodable.embed_feature_points(observed_arr)

        self.__mu_arr = self.__extractable_centroids.extract_centroids(feature_arr, self.__k)

    def learn(
        self, 
        np.ndarray observed_arr, 
        np.ndarray target_arr=None, 
        supervised_target_mode=False
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of noised observed data points.
        '''
        cdef np.ndarray noised_observed_arr
        if self.__denoising_flag is True:
            noised_observed_arr = observed_arr + self.__params_initializer.sample(
                size=observed_arr.copy().shape,
                **self.__params_dict
            )
        else:
            noised_observed_arr = observed_arr

        self.__auto_encodable.pre_learn(noised_observed_arr, observed_arr)

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
        try:
            loss_list = []
            for epoch in range(self.__epochs):
                self.__auto_encodable.inferencing_mode = False
                self.__opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                if target_arr is not None:
                    batch_target_arr = train_target_arr[rand_index]
                else:
                    batch_target_arr = None

                try:
                    q_arr = self.inference(batch_observed_arr)
                    if supervised_target_mode is False:
                        if epoch == 0 or epoch % self.__T == 0:
                            p_arr = self.compute_target_distribution(q_arr)
                    else:
                        if batch_target_arr is None:
                            raise ValueError("The `target_arr` must be not `None`.")
                        p_arr = np.repeat(
                            np.expand_dims(batch_target_arr, axis=2), 
                            repeats=q_arr.shape[2], 
                            axis=2
                        )

                    q_arr = q_arr.copy()

                    loss = self.compute_loss(
                        q_arr,
                        p_arr,
                        batch_target_arr
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
                    if target_arr is not None:
                        test_batch_target_arr = test_target_arr[rand_index]
                    else:
                        test_batch_target_arr = None

                    test_q_arr = self.inference(
                        test_batch_observed_arr
                    )

                    if supervised_target_mode is False:
                        if epoch == 0 or epoch % self.__T == 0:
                            test_p_arr = self.compute_target_distribution(test_q_arr)
                    else:
                        if test_batch_target_arr is None:
                            raise ValueError("The `target_arr` must be not `None`.")
                        test_p_arr = np.repeat(
                            np.expand_dims(test_batch_target_arr, axis=2), 
                            repeats=test_q_arr.shape[2], 
                            axis=2
                        )

                    test_loss = self.compute_loss(
                        test_q_arr,
                        test_p_arr,
                        test_batch_target_arr
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
    
    def compute_loss(self, p_arr, q_arr, target_arr=None):
        '''
        Compute loss.

        Args:
            p_arr:      `np.ndarray` of result of soft assignment.
            q_arr:      `np.ndarray` of target distribution.
            target_arr: `np.ndarray` of labeled data.
        
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
        self.__delta_mu_arr = np.dot(self.__feature_arr.T, delta_mu_arr).T / self.__batch_size
        self.__delta_z_arr = self.__grad_clipping(self.__delta_z_arr)
        self.__delta_mu_arr = self.__grad_clipping(self.__delta_mu_arr)
        self.__delta_z_arr = self.__delta_z_arr * self.__soft_assign_weight
        self.__delta_mu_arr = self.__delta_mu_arr * self.__soft_assign_weight

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_pc_arr = np.zeros((
            self.__batch_size, 
            self.__batch_size, 
            self.__feature_arr.shape[-1]
        ))
        self.__delta_pc_arr = np.zeros_like(delta_pc_arr)[:, 0, :]
        if target_arr is not None:
            pc_arr = self.compute_pairwise_constraint(target_arr)
            for i in range(self.__batch_size):
                for j in range(self.__batch_size):
                    if i != j:
                        delta_pc_arr[i, j] = np.square(self.__feature_arr[i] - self.__feature_arr[j])

            delta_pc_arr = np.expand_dims(pc_arr, axis=-1) * delta_pc_arr

            n = delta_pc_arr.reshape((delta_pc_arr.shape[0], -1)).copy().shape[1]
            self.__delta_pc_arr = np.nansum(delta_pc_arr, axis=1) / n
            self.__delta_pc_arr = self.__grad_clipping(self.__delta_pc_arr)
            self.__delta_pc_arr = self.__delta_pc_arr * self.__pairwise_lambda

        cdef np.ndarray delta_encoder_arr = None
        cdef np.ndarray delta_decoder_arr = None
        cdef np.ndarray delta_centroid_arr = None
        cdef np.ndarray _delta_encoder_arr = None
        cdef np.ndarray _delta_decoder_arr = None
        cdef np.ndarray _delta_centroid_arr = None

        for computable_clustering_loss in self.__computable_clustering_loss_list:
            _delta_encoder_arr, _delta_decoder_arr, _delta_centroid_arr = computable_clustering_loss.compute_clustering_loss(
                observed_arr=self.__observed_arr, 
                reconstructed_arr=self.__pred_arr, 
                feature_arr=self.__feature_arr,
                delta_arr=self.__delta_arr, 
                q_arr=q_arr, 
                p_arr=p_arr, 
            )
            if _delta_encoder_arr is not None:
                if delta_encoder_arr is None:
                    delta_encoder_arr = _delta_encoder_arr
                else:
                    delta_encoder_arr = delta_encoder_arr + _delta_encoder_arr

            if _delta_decoder_arr is not None:
                if delta_decoder_arr is None:
                    delta_decoder_arr = _delta_decoder_arr
                else:
                    delta_decoder_arr = delta_decoder_arr + _delta_decoder_arr

            if _delta_centroid_arr is not None:
                if delta_centroid_arr is None:
                    delta_centroid_arr = _delta_centroid_arr
                else:
                    delta_centroid_arr = delta_centroid_arr + _delta_centroid_arr

        if delta_encoder_arr is not None:
            self.__delta_encoder_arr = delta_encoder_arr
            loss_encoder = np.abs(self.__delta_encoder_arr).mean()
        else:
            self.__delta_encoder_arr = None
            loss_encoder = 0.0

        if delta_decoder_arr is not None:
            self.__delta_decoder_arr = delta_decoder_arr
            loss_decoder = np.abs(self.__delta_decoder_arr).mean()
        else:
            self.__delta_decoder_arr = None
            loss_decoder = 0.0

        if delta_centroid_arr is not None:
            self.__delta_centroid_arr = delta_centroid_arr
            loss_centroid = np.abs(self.__delta_centroid_arr).mean()
        else:
            self.__delta_centroid_arr = None
            loss_centroid = 0.0

        loss_arr = np.array([
            np.abs(self.__delta_mu_arr).mean(), 
            np.abs(self.__delta_z_arr).mean(), 
            np.abs(self.__delta_pc_arr).mean(),
            loss_encoder, 
            loss_decoder,
            loss_centroid
        ])
        self.__loss_arr = loss_arr
        return loss_arr.mean()

    def compute_pairwise_constraint(self, np.ndarray target_arr):
        target_arr = target_arr.argmax(axis=1)
        target_arr = target_arr + target_arr.max()
        target_arr = np.expand_dims(target_arr, axis=1)
        target_arr = target_arr / 1.0
        cdef np.ndarray pc_arr = (np.dot(target_arr, target_arr.T) == np.square(np.dot(target_arr, np.ones_like(target_arr).T))).astype(int)
        #pc_arr[pc_arr == 0] = -1
        return pc_arr

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
        if self.__delta_decoder_arr is not None:
            self.__auto_encodable.backward_auto_encoder(
                self.__delta_decoder_arr.reshape((
                    self.__observed_arr.copy().shape
                )),
                encoder_only_flag=False
            )
        cdef np.ndarray delta_arr = self.__delta_z_arr.reshape(
            self.__default_shape
        )
        if self.__delta_encoder_arr is not None:
            delta_arr = delta_arr + self.__delta_encoder_arr.reshape(
                self.__default_shape
            )

        if self.__delta_pc_arr is not None:
            delta_arr = delta_arr + self.__delta_pc_arr.reshape(
                self.__default_shape
            )

        self.__auto_encodable.backward_auto_encoder(
            delta_arr
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
        cdef np.ndarray delta_arr = self.__delta_mu_arr
        if self.__delta_centroid_arr is not None:
            delta_arr = delta_arr + self.__delta_centroid_arr
        grads_list = [
            delta_arr
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
