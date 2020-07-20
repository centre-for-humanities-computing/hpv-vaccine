'''
Train and operate with PMI-SVD embeddings
'''
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import linalg
import umap

import src.utility.general as util


class PmiSvdEmbeddings:

    def __init__(self, texts):
        '''
        Args:
        - texts (str|list): text data to model
        '''
        # check if texts are in the right format
        self.texts = util.validate_input(texts)
        # print report
        print('found {} document(s) in texts'.format(len(self.texts)))

        # initialize fields for later
        self.id_vocab = {}
        self.tok_vocab = {}
        self.model = {}

    def make_vocabulary(self):
        '''
        Assigns an index to each word.

        Creates two vocabularies for translation:
        id_vocab: keys are words, values are indices
        tok_vocab: keys are indices, values are words
        '''
        id_vocab = dict()
        for doc in self.texts:
            for token in doc:
                if token not in id_vocab:
                    id_vocab[token] = len(id_vocab)

        tok_vocab = {indx: tok for tok, indx in id_vocab.items()}

        # report
        print('found {} unique tokens in texts'.format(len(id_vocab)))

        # save dictionaries
        self.id_vocab = id_vocab
        self.tok_vocab = tok_vocab

    def count_pairs(self, back_window: int, front_window: int):
        '''
        Count times word pairs appear together in a specified window.

        Args:
        - back_window (int): how many words before should be counted as a pair
        - front_window (int): how many words after should be counted as a pair

        Returns:
        - pair_counts (collections.Counter): counts of word pairs
        '''
        # check input
        if not all([isinstance(back_window, int),
                    isinstance(front_window, int)]):
            raise ValueError("back_window and front_window must be int")

        # count pairs with counter
        pair_counts = Counter()
        for doc in self.texts:
            # indside docs, go through words
            for i_left, word in enumerate(doc):
                # minimum word index in the window
                lwr_window_limit = max(0, i_left - back_window)
                # maximum word index in the window
                upr_window_limit = min(len(doc) - 1, i_left + front_window)
                # get indices of words appearing
                # together with current word
                words_in_window = [i_right
                                   for i_right in range(lwr_window_limit,
                                                        upr_window_limit + 1)
                                   if i_right != i_left]

                # add word pairs to the counter
                for i_right in words_in_window:
                    skipgram = (doc[i_left], doc[i_right])
                    pair_counts[skipgram] += 1

        return pair_counts

    @staticmethod
    def sparse_it(row_ids, col_ids, values) -> np.array:
        '''
        transfrom lists into sparse np.array
        '''
        # determine shape (max index)
        this_shape = (max(row_ids)+1, max(col_ids)+1)
        # generate a matrix of zeroes of that thape
        sparse_mat = np.zeros(this_shape)
        # fill desired cells with values
        sparse_mat[row_ids, col_ids] = values

        return sparse_mat

    def word_cooc_matrix(self, pair_counts) -> np.array:
        '''
        Convert pair_counts into a sparse matrix.
        '''
        row_ids = list()
        col_ids = list()
        count_values = list()
        for (word_left, word_right), count in pair_counts.items():
            # find indices of word pairs
            word_left_i = self.id_vocab[word_left]
            word_right_i = self.id_vocab[word_right]
            # update lists
            row_ids.append(word_left_i)
            col_ids.append(word_right_i)
            count_values.append(count)

        cooc_mat = self.sparse_it(row_ids, col_ids, count_values)

        return cooc_mat

    def pmi(self,
            cooc_mat, id_vocab, pair_counts,
            alpha: float, pmi_type: str):
        '''
        Calculate Pointwise Mutual Information Measures.

        - Positive Pointwise Mutual Information (pmi_type='ppmi')
        - Smoothed Positive Pointwise Mutual Information (pmi_type='sppmi')
        '''

        # sums
        n_pairs = cooc_mat.sum()
        sum_over_words = cooc_mat.sum(axis=0).flatten()
        sum_over_contexts = cooc_mat.sum(axis=1).flatten()
        # smoothing
        sum_over_words_alpha = sum_over_words**alpha
        nca_denom = np.sum(cooc_mat.sum(axis=0).flatten()**alpha)

        # calculate PMI for each pair of words
        row_ids = []
        col_ids = []
        spmi_values = []
        sppmi_values = []
        for (word_left, word_right), count in pair_counts.items():
            # find indices of word pairs
            word_left_i = id_vocab[word_left]
            word_right_i = id_vocab[word_right]

            # define variables
            nwc = count
            Pwc = nwc / n_pairs
            nw = sum_over_contexts[word_left_i]
            Pw = nw / n_pairs
            nca = sum_over_words_alpha[word_right_i]
            Pca = nca / nca_denom

            # apply formula
            spmi = np.log2(Pwc/(Pw*Pca))
            # replace negative values with 0
            # by defualt, that is something you want
            sppmi = max(spmi, 0)

            # update lists
            row_ids.append(word_left_i)
            col_ids.append(word_right_i)
            spmi_values.append(spmi)
            sppmi_values.append(sppmi)

        # prepare sparse matrices for output
        if pmi_type == 'spmi':
            out_mat = self.sparse_it(row_ids, col_ids, spmi_values)
        if pmi_type == 'sspmi':
            out_mat = self.sparse_it(row_ids, col_ids, sppmi_values)

        return out_mat

    def train_model(self,
                    back_window: int, front_window: int,
                    pmi_type: str, alpha: float,
                    embedding_dim: int):
        '''
        Main method for fitting the PMI-SVD embeddings.

        Parameters
        ----------
        back_window : int
            How many words before x should be counted as a pair?

        front_window : int
            How many words after x should be counted as a pair

        pmi_type : str
            Type of PMI measure to use ('spmi' or 'sspmi')

        alpha : float
            Smoothing factor to apply to PMI scores. 
            Recommended alpha = 0.75

        embedding_dim : int
            How many dimensions word vectors should have?
        '''
        # create vocabularies
        self.make_vocabulary()
        # count pairs
        pair_counts = self.count_pairs(back_window, front_window)
        # make a matrix of word co-occurance
        cooc_mat = self.word_cooc_matrix(pair_counts)
        # calcualte Pointwise Mutual Information
        pmi_mat = self.pmi(cooc_mat=cooc_mat,
                           id_vocab=self.id_vocab,
                           pair_counts=pair_counts,
                           alpha=alpha,
                           pmi_type=pmi_type)

        # test if desired embedding_dim is possible to fit
        if embedding_dim >= pmi_mat.shape[1]:
            raise ValueError(
                'embedding_dim must be lower than \
                the number of columns in PMI matrix ({})'
                .format(pmi_mat.shape[1]))

        # reduce dimensions: Singular Value Decomposition
        u, s, vt = linalg.svds(pmi_mat, embedding_dim)
        self.model = u + vt.T

    def words_df(self) -> pd.DataFrame:
        '''
        Get a DF counting occurance of individual words.

        returns a dataframe of words and counts in descending order
        '''
        word_counts = Counter()
        for doc in self.texts:
            for token in doc:
                word_counts[token] += 1

        counter_df = (pd.Series(word_counts, name='count')
                      .reset_index()
                      .rename(columns={'index': 'token'})
                      .sort_values(by='count', ascending=False)
                      .reset_index())

        return counter_df

    def find(self, query):
        '''
        input word, get vector representation
        '''
        if query in self.id_vocab:
            query_index = self.id_vocab[query]
            return self.model[query_index]

        raise ValueError(
            '"{}" is not present in the model'.format(query))

    def dotprod_two_words(self, word1, word2):
        '''
        Calcualte the dot product between two given embeddings.

        The order doesn't matter.
        - word1: str, first word to query
        - word2: str, second word to query
        '''

        if not all(isinstance(i, str) for i in [word1, word2]):
            raise ValueError(
                f"expected string, please input a string to query")

        return np.dot(self.find(word1), self.find(word2))

    def similar_to(self, x, n_similar=5):
        '''
        Print n most similar words to x.

        Cosine similarity is the metric used (min = -1, max = 1)

        Args:
        - x (np.array | string): can be a vector, or a word that's in the model
        - n_similar: int, number of most simialr words to output
        '''

        if not isinstance(n_similar, int):
            raise ValueError(f"n_similar expected int")

        if isinstance(x, str):
            x = self.find(x)

        # cosine similarity matrix
        cos_sim_matrix = np.dot(self.model, x) / \
            (np.linalg.norm(self.model)*np.linalg.norm(x))

        # get indices of interesting cells
        similar_subset = np.argpartition(-1 * cos_sim_matrix,
                                         n_similar + 1)[:n_similar + 1]

        # extract similar words and score
        list_similar = [(float(cos_sim_matrix[i]), self.tok_vocab[i])
                        for i in similar_subset]

        return sorted(list_similar, reverse=True)

    def reduce_dim_umap(self,
                        n_components=2, n_neighbors=15,
                        min_dist=0.1, metric='cosine'):
        '''
        reduce the dimensions of tranined embeddings using UMAP.

        Output is 2D for easy visualization.
        Algorithm parameters are pre-defined here to give a drity overview
        of the global structure of the word-embedding model.

        Results may vary quite a bit based on UMAP paramenters.
        For tweaking the parameters, see
        https://umap-learn.readthedocs.io/en/latest/parameters.html
        '''

        reduced = umap.UMAP(
            # reduce to 2 dimensions
            n_components=n_components,
            # preserve local structure (low number)
            # or global structure (high number) of the data
            n_neighbors=n_neighbors,
            # how tightly do we alow to pack points together
            min_dist=min_dist,
            # correlational metric
            metric=metric)

        self.modelumap = reduced.fit_transform(self.model)
