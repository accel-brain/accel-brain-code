#!/user/bin/env python
# -*- coding: utf-8 -*-


class Ngram(object):
    '''
    N-gram
    '''

    def generate_ngram_data_set(self, token_list, n=2):
        '''
        Generate the N-gram's pair.

        Args:
            token_list:     The list of tokens.
            n               N

        Returns:
            zip of Tuple(Training N-gram data, Target N-gram data)
        '''
        n_gram_tuple_zip = self.generate_tuple_zip(token_list, n)
        n_gram_tuple_list = [n_gram_tuple for n_gram_tuple in n_gram_tuple_zip]
        n_gram_data_set = self.generate_tuple_zip(n_gram_tuple_list, 2)
        return n_gram_data_set

    def generate_skip_gram_data_set(self, token_list):
        '''
        Generate the Skip-gram's pair.

        Args:
            token_list:     The list of tokens.

        Returns:
            zip of Tuple(Training N-gram data, Target N-gram data)
        '''
        n_gram_tuple_zip = self.generate_tuple_zip(token_list, 3)
        skip_gram_list = []
        for pre, point, post in n_gram_tuple_zip:
            skip_gram_list.append((point, pre))
            skip_gram_list.append((point, post))
        return zip(skip_gram_list)

    def generate_tuple_zip(self, token_list, n=2):
        '''
        Generate the N-gram.

        Args:
            token_list:     The list of tokens.
            n               N

        Returns:
            zip of Tuple(N-gram)
        '''
        return zip(*[token_list[i:] for i in range(n)])
