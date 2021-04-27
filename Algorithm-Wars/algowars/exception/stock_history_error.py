#!/user/bin/env python
# -*- coding: utf-8 -*-


class StockHistoryError(Exception):
    '''
    '''
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return repr(self.value)
