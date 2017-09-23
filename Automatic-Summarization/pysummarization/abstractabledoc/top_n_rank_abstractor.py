# -*- coding: utf-8 -*-
from pysummarization.abstractable_doc import AbstractableDoc


class TopNRankAbstractor(AbstractableDoc):
    '''
    Ranking the list of tokens.
    '''

    # N of top-n.
    __top_n = 10

    def get_top_n(self):
        ''' getter '''
        if isinstance(self.__top_n, int) is False:
            raise TypeError("The type of __top_n must be int.")
        return self.__top_n

    def set_top_n(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __top_n must be int.")
        self.__top_n = value

    top_n = property(get_top_n, set_top_n)

    def filter(self, scored_list):
        '''
        Filtering with top-n ranking.

        Args:
            scored_list:    The list of scoring.

        Retruns:
            The list of filtered result.

        '''
        top_n_key = -1 * self.top_n
        top_n_list = sorted(scored_list, key=lambda x: x[1])[top_n_key:]
        result_list = sorted(top_n_list, key=lambda x: x[0])
        return result_list
