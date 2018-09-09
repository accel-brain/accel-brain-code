# -*- coding: utf-8 -*-
from pysummarization.nlp_base import NlpBase
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.abstractabledoc.std_abstractor import StdAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.readablewebpdf.web_pdf_reading import WebPDFReading
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
from pysummarization.similarityfilter.dice import Dice
from pysummarization.similarityfilter.jaccard import Jaccard
from pysummarization.similarityfilter.simpson import Simpson
from pysummarization.similarityfilter.encoder_decoder_cosine import EncoderDecoderCosine
from pysummarization.similarityfilter.lstm_rtrbm_cosine import LSTMRTRBMCosine
from pysummarization.similarityfilter.lstm_rtrbm_clustering import LSTMRTRBMClustering
from pysummarization.clusterabledoc.k_means import KMeans


def Main(url, similarity_mode="TfIdfCosine", similarity_limit=0.75):
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

    if similarity_mode == "EncoderDecoderCosine":
        # The object of `Similarity Filter`.
        # The similarity observed by this object is so-called cosine similarity of manifolds,
        # which is embedded in hidden layer of Encoder/Decoder based on LSTM.
        similarity_filter = EncoderDecoderCosine(
            document,
            hidden_neuron_count=200,
            epochs=100,
            batch_size=100,
            learning_rate=1e-05,
            learning_attenuate_rate=0.1,
            attenuate_epoch=50,
            bptt_tau=8,
            weight_limit=0.5,
            dropout_rate=0.5,
            test_size_rate=0.3,
            debug_mode=True
        )

    elif similarity_mode == "LSTMRTRBMCosine":
        # The object of `Similarity Filter`.
        # The similarity observed by this object is so-called cosine similarity of manifolds,
        # which is embedded in hidden layer of LSTM-RTRBM.
        similarity_filter = LSTMRTRBMCosine(
            document,
            training_count=1,
            hidden_neuron_count=1000,
            batch_size=100,
            learning_rate=1e-03,
            seq_len=5,
            debug_mode=True
        )
    
    elif similarity_mode == "KMeans":
        # The object of `Similarity Filter`.
        # The similarity is observed by checking whether each sentence belonging to the same cluster, 
        # and if so, the similarity is `1.0`, if not, the value is `0.0`.
        # The data clustering algorithm is based on K-Means method, 
        # learning data which is embedded in hidden layer of LSTM-RTRBM.
        similarity_filter = LSTMRTRBMClustering(
            document,
            clusterable_doc=KMeans(cluster_num=10, max_iter=100),
            tokenizable_doc=None,
            hidden_neuron_count=1000,
            batch_size=100,
            learning_rate=1e-03,
            seq_len=5,
            debug_mode=True
        )

    elif similarity_mode == "TfIdfCosine":
        # The object of `Similarity Filter`. 
        # The similarity observed by this object is so-called cosine similarity of Tf-Idf vectors.
        similarity_filter = TfIdfCosine()

    elif similarity_mode == "Dice":
        # The object of `Similarity Filter`.
        # The similarity observed by this object is the Dice coefficient.
        similarity_filter = Dice()

    elif similarity_mode == "Jaccard":
        # The object of `Similarity Filter`.
        # The similarity observed by this object is the Jaccard coefficient.
        similarity_filter = Jaccard()
    
    elif similarity_mode == "Simpson":
        # The object of `Similarity Filter`.
        # The similarity observed by this object is the Simpson coefficient.
        similarity_filter = Simpson()
    
    else:
        raise ValueError()


    # The object of the NLP.
    nlp_base = NlpBase()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    nlp_base.tokenizable_doc = MeCabTokenizer()
    # Set the object of NLP.
    similarity_filter.nlp_base = nlp_base
    # If the similarity exceeds this value, the sentence will be cut off.
    similarity_filter.similarity_limit = similarity_limit

    # The object of automatic sumamrization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    auto_abstractor.tokenizable_doc = MeCabTokenizer()
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Execute summarization.
    result_dict = auto_abstractor.summarize(document, abstractable_doc, similarity_filter)
    # Output summarized sentence.
    [print(result_dict["summarize_result"][i]) for i in range(len(result_dict["summarize_result"])) if i < 3]

if __name__ == "__main__":
    import sys
    # PDF url.
    url = sys.argv[1]
    if len(sys.argv) > 2:
        similarity_mode = sys.argv[2]
    else:
        similarity_mode = "TfIdfCosine"
    if len(sys.argv) > 3:
        similarity_limit = float(sys.argv[3])
    else:
        similarity_limit = 0.75

    Main(url, similarity_mode, similarity_limit)
