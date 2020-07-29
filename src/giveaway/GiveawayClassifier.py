'''
Multinomial Bayes for text classification.

In this project,
binary classification of FB posts (detecting giveaways & viral marketing)

TODO:
- docstrings
'''

import re
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, precision_score, recall_score


class GiveawayClassifier:

    def __init__(self, X, y, seed = 11):
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.3,
                                                                    random_state=seed)
        
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw
        self.y_train = y_train
        self.y_test = y_test
        
    def _make_features(self):
        
        # TODO: cut off at certain word frequency
        
        # prepare vocabulary
        texts = self.X_train_raw.tolist()
        texts = [doc.split() for doc in texts]

        tok2indx = dict()
        unigram_counts = Counter()
        for ii, text in enumerate(texts):
            for token in text:
                unigram_counts[token] += 1
                if token not in tok2indx:
                    tok2indx[token] = len(tok2indx)
        indx2tok = {indx: tok for tok, indx in tok2indx.items()}
        
        
        # occurance counts
        vectorizer_train = CountVectorizer(vocabulary=tok2indx)
        X_train = vectorizer_train.fit_transform(self.X_train_raw)
        
        vectorizer_test = CountVectorizer(vocabulary=vectorizer_train.vocabulary_)
        X_test = vectorizer_test.transform(self.X_test_raw)
        
        self.X_train = X_train
        self.X_test = X_test
        self.vocab = tok2indx

    def evaluate_model(self, y_true, y_pred):

        # df to save resutls to
        out = dict()

        # confusion matrix based measures
        ac = accuracy_score(y_true, y_pred)

        brG = brier_score_loss(y_true, y_pred, pos_label=0)
        brB = brier_score_loss(y_true, y_pred, pos_label=1)
        recG = recall_score(y_true, y_pred, pos_label=0)
        recB = recall_score(y_true, y_pred, pos_label=1)
        preG = precision_score(y_true, y_pred, pos_label=0)
        preB = precision_score(y_true, y_pred, pos_label=1)

        # append to out dict
        out.update({"accuracy": [ac]})
        out.update({"brier_n": [brG]})
        out.update({"brier_giveaway": [brB]})
        out.update({"recall_n": [recG]})
        out.update({"recall_giveaway": [recB]})
        out.update({"precision_n": [preG]})
        out.update({"precision_giveaway": [preB]})

        return pd.DataFrame.from_dict(out)

    def train(self):
        
        # preprocess
        self._make_features()
        
        # fit model
        self.model = MultinomialNB().fit(self.X_train, self.y_train)
        
        # make predictions
        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)       
        
        # evaluate
        report_train = self.evaluate_model(y_true=self.y_train, y_pred=pred_train)
        report_train.index = ['train']
        report_test = self.evaluate_model(y_true=self.y_test, y_pred=pred_test)
        report_test.index = ['test']
        
        # accuracy report
        self.report = pd.concat([report_train, report_test])
        
        # close reading of input posts
        ## for train
        df_train = pd.DataFrame(self.X_train_raw)
        ser_train_y_real = pd.Series(self.y_train, name = "actual")
        ser_train_y_pred = pd.Series(pred_train, name = "predicted")
        
        train_aug = (
            df_train
            .join(ser_train_y_real)
            .reset_index()
            .join(ser_train_y_pred)
        )
        train_aug['split'] = 'train'
        
        ## for test
        df_test = pd.DataFrame(self.X_test_raw)
        ser_test_y_real = pd.Series(self.y_test, name = 'actual')
        ser_test_y_pred = pd.Series(pred_test, name = 'predicted')

        test_aug = (
            df_test
            .join(ser_test_y_real)
            .reset_index()
            .join(ser_test_y_pred)
        )
        test_aug['split'] = 'test'
        
        self.df_labeled = pd.concat([train_aug, test_aug])
        # per term probability
        terms = pd.DataFrame(list(self.vocab.items()), columns=['token', 'id'])
        prob_neutral = pd.Series(self.model.feature_log_prob_[0],
                                 name='prob_neutral')
        prob_giveaway = pd.Series(self.model.feature_log_prob_[1],
                                  name='prob_giveaway')

        self.term_importance = terms.join(prob_neutral).join(prob_giveaway)
    
    def get_misses(self, which=None, split=None, report=False):
        
        # get the right split
        if split == "train":
            if report: print("Train:")
            d = self.df_labeled.query('split == "train"')
            
        elif split == "test":
            if report: print("Test:")
            d = self.df_labeled.query('split == "test"')
            
        else:
            if report: print("All data:")
            d = self.df_labeled
        
        # get false positives or false negatives
        if which == 'false_positive' or which == "fp":
            if report: print('False Positives')
            d = d.query('actual == 0').query('predicted == 1')
            
        elif which == 'false_negative' or which == "fn":
            if report: print('False Negatives')
            d = d.query('actual == 1').query('predicted == 0')
            
        else:
            if report: print('displaying all misses')
            pass
        
        return d

    @staticmethod
    def _negative_for_url(df):
        '''
        A post that only contains a link should not be classified as a giveaway.
        This rewrites the model's prediction for texts that are a link.
        '''
        # a monster regex for matching links
        url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')

        url_idx = np.array(df['text'].str.match(url_pattern))
        df.loc[url_idx, ['predicted']] = 0

        return df

    def predict_new(self, new_data, negative_for_url=False):

        vectorizer_new = CountVectorizer(vocabulary=self.vocab)
        X_new = vectorizer_new.transform(new_data)

        pred_new = self.model.predict(X_new)
        ser_new_y_pred = pd.Series(pred_new, name = "predicted")

        new_df = pd.DataFrame(new_data)
        new_aug = new_df.reset_index().join(ser_new_y_pred)

        if negative_for_url:
            new_aug = self._negative_for_url(new_aug)

        return new_aug
