# -*- coding: utf-8 -*-
from accelbrainbase.iteratabledata._mxnet.ssda_iterator import SSDAIterator
import mxnet.ndarray as nd
import numpy as np
from abc import abstractmethod


class CroppingRotationIterator(SSDAIterator):
    '''
    Iterator that draws from image files and generates `mxnet.ndarray`.

    References:
        - Jing, L., & Tian, Y. (2020). Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.
        - Xu, J., Xiao, L., & López, A. M. (2019). Self-supervised domain adaptation for computer vision tasks. IEEE Access, 7, 156694-156706., p156698.
    '''

    def create_pretext_task_samples(self, target_domain_batch_arr):
        '''
        Create samples for pretext_task.

        Args:
            target_domain_batch_arr:    `nd.ndarray` of samples in target domain.
        
        Returns:
            Tuple data. The shape is ...
            - pretext_task samples.
            - pretext_task labels.
        '''
        pretext_arr_1 = target_domain_batch_arr[:, :, :target_domain_batch_arr.shape[2]//2, :target_domain_batch_arr.shape[3]//2]
        pretext_arr_2 = target_domain_batch_arr[:, :, target_domain_batch_arr.shape[2]//2:, :target_domain_batch_arr.shape[3]//2]
        pretext_arr_3 = target_domain_batch_arr[:, :, :target_domain_batch_arr.shape[2]//2, target_domain_batch_arr.shape[3]//2:]
        pretext_arr_4 = target_domain_batch_arr[:, :, target_domain_batch_arr.shape[2]//2:, target_domain_batch_arr.shape[3]//2:]

        angle_list = []
        for batch in range(target_domain_batch_arr.shape[0]):
            angle_key = np.random.randint(low=0, high=4)
            angle_arr_1 = np.zeros(4)
            angle_arr_1[angle_key] = 1
            angle = 90 * angle_key
            pretext_arr_1[batch] = self.__rotate(pretext_arr_1[batch], angle)

            angle_key = np.random.randint(low=0, high=4)
            angle_arr_2 = np.zeros(4)
            angle_arr_2[angle_key] = 1
            angle = 90 * angle_key
            pretext_arr_2[batch] = self.__rotate(pretext_arr_2[batch], angle)

            angle_key = np.random.randint(low=0, high=4)
            angle_arr_3 = np.zeros(4)
            angle_arr_3[angle_key] = 1
            angle = 90 * angle_key
            pretext_arr_3[batch] = self.__rotate(pretext_arr_3[batch], angle)

            angle_key = np.random.randint(low=0, high=4)
            angle_arr_4 = np.zeros(4)
            angle_arr_4[angle_key] = 1
            angle = 90 * angle_key
            pretext_arr_4[batch] = self.__rotate(pretext_arr_4[batch], angle)

            angle_list.append(
                np.r_[
                    angle_arr_1,
                    angle_arr_2,
                    angle_arr_3,
                    angle_arr_4
                ]
            )

        pretext_arr = nd.concat(
            nd.concat(pretext_arr_1, pretext_arr_2, dim=2),
            nd.concat(pretext_arr_3, pretext_arr_4, dim=2),
            dim=3
        )
        angle_arr = nd.ndarray.array(angle_list, ctx=pretext_arr.context)

        return pretext_arr, angle_arr

    def __rotate(self, arr, angle=0):
        if angle == 0:
            return arr
        elif angle == 90:
            arr = arr.transpose((0, 2, 1))
            return arr
        elif angle == 180:
            arr = nd.flip(arr, axis=1)
            return arr
        elif angle == 270:
            arr = arr.transpose((0, 2, 1))
            arr = nd.flip(arr, axis=1)
            return arr
