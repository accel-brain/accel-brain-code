# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.abstractable_doc import AbstractableDoc


class StdAbstractor(AbstractableDoc):
    '''
    The filtering list of tokens with the difference between the standard deviation.
    '''

    def filter(self, scored_list):
        '''
        Filtering with std.

        Args:
            scored_list:    The list of scoring.

        Retruns:
            The list of filtered result.

        '''
        if len(scored_list) > 0:
            avg = np.mean([s[1] for s in scored_list])
            std = np.std([s[1] for s in scored_list])
        else:
            avg = 0
            std = 0
        limiter = avg + 0.5 * std
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_list if score > limiter]
        return mean_scored
