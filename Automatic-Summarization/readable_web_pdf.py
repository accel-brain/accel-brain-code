#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ReadableWebPDF(metaclass=ABCMeta):
    '''
    Read strings in PDF documents
    '''

    @abstractmethod
    def url_to_text(self, url):
        '''
        Transform PDF documents to strings.

        Args:
            url:   URL

        Returns:
            string.

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def is_pdf_url(self, url):
        '''
        Check PDF format.

        Args:
            url:    URL

        Returns:
            True: PDF, False: not PDF
        '''
        raise NotImplementedError("This method must be implemented.")
