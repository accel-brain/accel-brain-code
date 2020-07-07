# -*- coding: utf-8 -*-
from accelbrainbase.extractable_data import ExtractableData
import pandas as pd


class UnlabeledCSVExtractor(ExtractableData):
    '''
    CSV Extractor for unlabeled samples.
    '''

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
        return df.values
