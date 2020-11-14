# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData


class SSDAIterator(IteratableData):
    '''
    Iterator that draws from image files and generates the samples.

    References:
        - Ghifary, M., Kleijn, W. B., Zhang, M., Balduzzi, D., & Li, W. (2016, October). Deep reconstruction-classification networks for unsupervised domain adaptation. In European Conference on Computer Vision (pp. 597-613). Springer, Cham.

    '''

    def pre_normalize(self, arr):
        '''
        Normalize before observation.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        if self.__norm_mode == "min_max":
            if arr.max() != arr.min():
                n = 0.0
            else:
                n = 1e-08
            arr = (arr - arr.min()) / (arr.max() - arr.min() + n)
        elif self.__norm_mode == "z_score":
            std = arr.asnumpy().std()
            if std == 0:
                std += 1e-08
            arr = (arr - arr.mean()) / std

        arr = arr * self.__scale
        return arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_epochs(self):
        ''' getter '''
        return self.__epochs

    def set_epochs(self, value):
        ''' setter '''
        self.__epochs = value

    epochs = property(get_epochs, set_epochs)

    def get_batch_size(self):
        ''' getter '''
        return self.__batch_size

    def set_batch_size(self, value):
        ''' setter '''
        self.__batch_size = value

    batch_size = property(get_batch_size, set_batch_size)

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value

    seq_len = property(get_seq_len, set_seq_len)

    __norm_mode = "z_score"

    def get_norm_mode(self):
        ''' getter '''
        return self.__norm_mode
    
    def set_norm_mode(self, value):
        ''' setter '''
        self.__norm_mode = value
    
    norm_mode = property(get_norm_mode, set_norm_mode)

    __scale = 1.0

    def get_scale(self):
        ''' getter '''
        return self.__scale
    
    def set_scale(self, value):
        ''' setter '''
        self.__scale = value
    
    scale = property(get_scale, set_scale)
