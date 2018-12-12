# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class AbstractableDoc(metaclass=ABCMeta):
    '''
    Automatic abstraction and summarization.
    This is the filtering approach.

    This `interface` is designed the `Strategy Pattern`.

    References:
        - Luhn, Hans Peter. "The automatic creation of literature abstracts."  IBM Journal of research and development 2.2 (1958): 159-165.
        - http://www.oreilly.co.jp/books/9784873116792/

    '''

    @abstractmethod
    def filter(self, scored_list):
        '''
        Execute filtering sentences.

        Args:
            scored_list:    The list of statistical information derived from word frequency and distribution.

        Retruns:
            the list of filtered sentence.

        '''
        raise NotImplementedError("This method must be implemented.")
