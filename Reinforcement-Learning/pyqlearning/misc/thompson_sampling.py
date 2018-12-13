#!/user/bin/env python
# -*- coding: utf-8 -*-
from beta_dist import BetaDist


class ThompsonSampling(object):
    '''
    Thompson Sampling.
    '''

    # Dict of Beta Distribution.
    __beta_dist_dict = {}

    def __init__(self, arm_id_list):
        '''
        Initialization

        Args:
            arm_id_list:    List of arms Master id.
        '''
        [self.__beta_dist_dict.setdefault(key, BetaDist()) for key in arm_id_list]

    def pull(self, arm_id, success, failure):
        '''
        Pull arms.

        Args:
            arm_id:     Arms master id.
            success:    The number of success.
            failure:    The number of failure.
        '''
        self.__beta_dist_dict[arm_id].observe(success, failure)

    def recommend(self, limit=10):
        '''
        Listup arms and expected value.

        Args:
            limit:      Length of the list.

        Returns:
            [Tuple(`Arms master id`, `expected value`)]
        '''
        expected_list = [(arm_id, beta_dist.expected_value()) for arm_id, beta_dist in self.__beta_dist_dict.items()]
        expected_list = sorted(expected_list, key=lambda x: x[1], reverse=True)
        return expected_list[:limit]
