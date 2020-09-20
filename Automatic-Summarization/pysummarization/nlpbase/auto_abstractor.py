import nltk
from pysummarization.nlp_base import NlpBase
from pysummarization.abstractable_doc import AbstractableDoc
from pysummarization.similarity_filter import SimilarityFilter

class AutoAbstractor(NlpBase):
    '''
    The object for automatic summarization.
    '''

    # Only top-n scored tokens must be set to target. 
    __target_n = 100

    def get_target_n(self):
        ''' getter '''
        if isinstance(self.__target_n, int) is False:
            raise TypeError("The type of __target_n must be int.")
        return self.__target_n

    def set_target_n(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __target_n must be int.")
        self.__target_n = value

    target_n = property(get_target_n, set_target_n)

    # Adjacent distance.
    __cluster_threshold = 5

    def get_cluster_threshold(self):
        ''' getter '''
        if isinstance(self.__cluster_threshold, int) is False:
            raise TypeError("The type of __cluster_threshold must be int.")
        return self.__cluster_threshold

    def set_cluster_threshold(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __cluster_threshold must be int.")
        self.__cluster_threshold = value

    cluster_threshold = property(get_cluster_threshold, set_cluster_threshold)

    # The number of returned sentences.
    __top_sentences = 5

    def get_top_sentences(self):
        ''' getter '''
        if isinstance(self.__top_sentences, int) is False:
            raise TypeError("The type of __top_sentences must be int.")
        return self.__top_sentences

    def set_top_sentences(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __top_sentences must be int.")
        self.__top_sentences = value

    top_sentences = property(get_top_sentences, set_top_sentences)

    def summarize(self, document, Abstractor, similarity_filter=None):
        '''
        Execute summarization.

        Args:
            document:           The target document.
            Abstractor:         The object of AbstractableDoc.
            similarity_filter   The object of SimilarityFilter.

        Returns:
            dict data.
            - "summarize_result": The list of summarized sentences., 
            - "scoring_data":     The list of scores.
        '''
        if isinstance(document, str) is False:
            raise TypeError("The type of document must be str.")

        if isinstance(Abstractor, AbstractableDoc) is False:
            raise TypeError("The type of Abstractor must be AbstractableDoc.")

        if isinstance(similarity_filter, SimilarityFilter) is False and similarity_filter is not None:
            raise TypeError("The type of similarity_filter must be SimilarityFilter.")

        normalized_sentences = self.listup_sentence(document)

        # for filtering similar sentences.
        if similarity_filter is not None:
            normalized_sentences = similarity_filter.similar_filter_r(normalized_sentences)

        self.tokenize(document)
        words = self.token

        fdist = nltk.FreqDist(words)
        top_n_words = [w[0] for w in fdist.items()][:self.target_n]
        scored_list = self.__closely_associated_score(normalized_sentences, top_n_words)
        filtered_list = Abstractor.filter(scored_list)
        result_list = [normalized_sentences[idx] for (idx, score) in filtered_list]
        result_dict = {
            "summarize_result": result_list,
            "scoring_data": filtered_list
        }
        return result_dict

    def __closely_associated_score(self, normalized_sentences, top_n_words):
        '''
        Scoring the sentence with closely associations.

        Args:
            normalized_sentences:   The list of sentences.
            top_n_words:            Important sentences.

        Returns:
            The list of scores.
        '''
        scores_list = []
        sentence_idx = -1

        for sentence in normalized_sentences:
            self.tokenize(sentence)
            sentence = self.token

            sentence_idx += 1
            word_idx = []

            for w in top_n_words:
                try:
                    word_idx.append(sentence.index(w))
                except ValueError:
                    pass

            word_idx.sort()

            if len(word_idx) == 0:
                continue

            clusters = []
            cluster = [word_idx[0]]
            i = 1
            while i < len(word_idx):
                if word_idx[i] - word_idx[i - 1] < self.cluster_threshold:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)

            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                score = 1.0 * significant_words_in_cluster \
                    * significant_words_in_cluster / total_words_in_cluster

                if score > max_cluster_score:
                    max_cluster_score = score

            scores_list.append((sentence_idx, max_cluster_score))

        return scores_list
