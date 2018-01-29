# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.similarity_filter import SimilarityFilter
from pysummarization.vectorlizabletoken.tfidf_vectorizer import TfidfVectorizer


class TiIdfCosine(SimilarityFilter):
    '''
    Concrete class for filtering mutually similar sentences.
    '''

    def calculate(self, token_list_x, token_list_y):
        '''
        Calculate similarity with the so-called Cosine similarity of Tf-Idf vectors.
        
        Concrete method.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]
        
        Returns:
            Similarity.
        '''

        document_list = token_list_x.copy()
        [document_list.append(v) for v in token_list_y]
        document_list = list(set(document_list))

        tfidf_vectorizer = TfidfVectorizer(document_list)

        vector_list_x = tfidf_vectorizer.vectorlize(token_list_x)
        vector_list_y = tfidf_vectorizer.vectorlize(token_list_y)
        
        if len(vector_list_x) > len(vector_list_y):
            [vector_list_y.append(0.0) for _ in range(len(vector_list_x) - len(vector_list_y))]
        elif len(vector_list_y) > len(vector_list_x):
            [vector_list_x.append(0.0) for _ in range(len(vector_list_y) - len(vector_list_x))]

        dot_prod = np.dot(vector_list_x, vector_list_y)
        norm_x = np.linalg.norm(vector_list_x)
        norm_y = np.linalg.norm(vector_list_y)
        try:
            return dot_prod / (norm_x * norm_y)
        except ZeroDivisionError:
            return 0.0
