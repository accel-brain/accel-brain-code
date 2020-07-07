# -*- coding: utf-8 -*-
from accelbrainbase.extractable_data import ExtractableData
import pandas as pd


class LabeledCSVExtractor(ExtractableData):
    '''
    CSV Extractor for labeled samples.
    '''

    # `str` of column of label
    __label_column = "label"

    def __init__(
        self, 
        label_column="label"
    ):
        '''
        Init.

        Args:
            label_column:        `str` of column of label.
        '''
        self.label_column = label_column

    def extract(
        self,
        path,
    ):
        '''
        Extract CSV file data.

        Args:
            path:     `str` of CSV files.
        
        Returns:
            Tuple data.
            - Observed data points.
            - label data.
        '''
        df = pd.read_csv(path)
        label_arr = df[self.label_column].values
        observed_arr = df[[col for col in df.columns if col != self.label_column]].values

        return observed_arr, label_arr

    def get_label_column(self):
        ''' getter of `str` of column of label. '''
        return self.__label_column

    def set_label_column(self, value):
        ''' setter of `str` of column of label. '''
        if isinstance(value, str):
            self.__label_column = value
        else:
            raise TypeError("The type of `label_column` must be `str`.")
    
    label_column = property(get_label_column, set_label_column)
