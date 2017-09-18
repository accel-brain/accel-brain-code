# -*- coding: utf-8 -*-
from nlpbase.auto_abstractor import AutoAbstractor
from n_gram import Ngram


class NgramAutoAbstractor(AutoAbstractor):
    '''
    文書の自動要約アルゴリズム。
    トークナイズの最小単位をn-gramにする
    '''
    
    # N-gramのオブジェクト
    __n_gram = None
    
    def get_n_gram(self):
        if isinstance(self.__n_gram, Ngram):
            return self.__n_gram
        else:
            raise TypeError("The type of n_gram must be Ngram.")

    def set_n_gram(self, value):
        if isinstance(value, Ngram):
            self.__n_gram = value
        else:
            raise TypeError("The type of n_gram must be Ngram.")

    n_gram = property(get_n_gram, set_n_gram)

    # N-gramのn
    __n = 2
    
    def get_n(self):
        if isinstance(self.__n, int):
            return self.__n
        else:
            raise TypeError("The type of n must be int.")

    def set_n(self, value):
        if isinstance(value, int):
            self.__n = value
        else:
            raise TypeError("The type of n must be int.")

    n = property(get_n, set_n)

    def tokenize(self, data):
        '''
        オーバーライド
        形態素解析して、トークンのn-gramをプロパティ：tokenにセットする。

        Args:
            形態素解析対象となる[n-gram, n-gram, ...]。

        '''
        super().tokenize(data)
        token_tuple_zip = self.n_gram.generate_tuple_zip(self.token, self.n)
        token_list = []
        self.token = ["".join(list(token_tuple)) for token_tuple in token_tuple_zip]
