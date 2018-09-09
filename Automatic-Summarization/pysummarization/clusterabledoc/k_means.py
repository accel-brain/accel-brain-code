#i -*- coding: utf-8 -*-
from pysummarization.clusterable_doc import ClusterableDoc
import numpy as np


class KMeans(ClusterableDoc):
    '''
    KMeans method.
    '''
    
    # Centroid of clusters.
    __centroid_arr = None
    
    # The number of cluseters.
    __cluster_num = 2
    
    # Maximum number of iterations.
    __max_iter = 100

    def __init__(self, cluster_num=2, max_iter=100):
        '''
        Init.

        Args:
            cluster_num:     The number of clusters.
            max_iter:        Maximum number of iterations.
        '''
        self.__cluster_num = cluster_num
        self.__max_iter = max_iter

    def learn(self, document_arr):
        '''
        Learning.

        Args:
            document_arr:    `np.ndarray` of sentences vectors.

        Retruns:
            `np.ndarray` of labeled data.

        '''
        self.__centroid_arr = np.random.normal(loc=document_arr.mean(), scale=document_arr.std(), size=document_arr.shape)
        inferenced_arr = np.zeros(document_arr.shape)
        for _ in range(self.__max_iter):
            inferenced_arr = np.array([np.array([self.__compute_distance(p_arr, c_arr) for c_arr in self.__centroid_arr]) for p_arr in document_arr]) 
            inferenced_arr = inferenced_arr.argmin(axis=1)
            centroid_arr = np.array([np.mean(document_arr[inferenced_arr == i], axis=0) for i in range(self.__cluster_num)])
            if np.nansum(centroid_arr) == 0:
                continue
            self.__centroid_arr = centroid_arr

        return inferenced_arr

    def inference(self, document_arr):
        '''
        Inferencing.

        Args:
            document_arr:    `np.ndarray` of sentences vectors.

        Retruns:
            `np.ndarray` of labeled data.
        '''
        inferenced_arr = np.array([np.array([self.__compute_distance(p_arr, c_arr) for c_arr in self.__centroid_arr]).argmin() for p_arr in document_arr])
        return inferenced_arr

    def __compute_distance(self, p_arr, c_arr):
        return np.nansum(np.square(p_arr - c_arr))

    def get_centroid_arr(self):
        ''' getter '''
        return self.__centroid_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    centroid_arr = property(get_centroid_arr, set_readonly)
