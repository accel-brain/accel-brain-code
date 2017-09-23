# -*- coding: utf-8 -*-
from nlpbase.auto_abstractor import AutoAbstractor
from tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from web_scraping import WebScraping
from abstractabledoc.std_abstractor import StdAbstractor
from abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from readablewebpdf.web_pdf_reading import WebPDFReading


def Main(url):
    '''
    Entry Point.
    
    Args:
        url:    PDF url.
    '''
    # The object of Web-scraping.
    web_scrape = WebScraping()
    # Set the object of reading PDF files.
    web_scrape.readable_web_pdf = WebPDFReading()
    # Execute Web-scraping.
    document = web_scrape.scrape(url)
    # The object of automatic sumamrization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    auto_abstractor.tokenizable_doc = MeCabTokenizer()
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Execute summarization.
    result_list = auto_abstractor.summarize(document, abstractable_doc)
    # Output summarized sentence.
    [print(sentence) for sentence in result_list["summarize_result"]]

if __name__ == "__main__":
    import sys
    # PDF url.
    url = sys.argv[1]
    Main(url)
