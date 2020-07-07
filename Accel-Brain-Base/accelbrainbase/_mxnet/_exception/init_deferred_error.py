# -*- coding: utf-8 -*-


class InitDeferredError(Exception):
    '''
    Exception, raised when the initialization should be deferred.
    '''

    def __init__(self, value=""):
        '''
        Init.

        Args:
            value:      The error message.
        '''
        self.value = value
    
    def __str__(self):
        return repr(self.value)
