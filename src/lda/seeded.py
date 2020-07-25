'''Iterate guidedlda's LDA in search of good hyperparameters.

TODO
- TfIdfVectorizer
- save priors

- iterate seed_confidence?
'''
import os
from itertools import chain
from time import time

import pickle
import ndjson

from sklearn.feature_extraction.text import CountVectorizer, TfIdfVectorizer
import guidedlda
import pyLDAvis.sklearn

from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel

from src.utility.general import make_folders


def init_guidedlda(texts, seed_topic_list, vectorizer_type='count'):
    '''
    Prepare data & transform seeds to priors for guidedlda.
    (Vectorize texts and extract hashed seeds)

    No more preprocessing is done here.


    Parameters
    ----------
    texts : iterable
        already preprocessed text data you want to build seeds on.

    seed_topic_list : list (of lists)
        list of words, where in seed_topic_list[x][y] 
        x is a topic and y a word belonging in that topic.

    vectorizer_type : str (optional)
        Map documents using raw counts, or tfidf?
        Options = {"count", "tfidf"}
    '''
    # texts are already preprocessed, so vectorizer gets this as tokenizer
    def do_nothing(doc):
        return doc

    # choose vectorizer
    if vectorizer_type is 'tfidf':
        # TODO: this is dummy code
        vectorizer = TfIdfVectorizer(
            analyzer='word',
            tokenizer=do_nothing,
            preprocessor=None,
            lowercase=False
        )

    elif vectorizer_type is 'count':
        vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=do_nothing,
            preprocessor=None,
            lowercase=False
        )

    else:
        raise ValueError('vectorizer_type: choose "tfidf", or "count"')

    # prep texts to guidedlda format
    X = vectorizer.fit_transform(texts)
    tf_feature_names = vectorizer.get_feature_names()
    word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))

    # catch seed-word hashes to give to the model
    seed_priors = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            try:
                seed_priors[word2id[word]] = t_id
            except KeyError:
                pass

    return X, seed_priors, vectorizer


def gensim_format(texts):
    '''
    Tokenized texts to gensim objects.
    For calculating topic coherence using gensim's CoherenceModel.
    '''

    dictionary = corpora.Dictionary(texts)
    bows = [dictionary.doc2bow(tl) for tl in texts]

    return bows, dictionary


def coherence_guidedlda(topics, bows, dictionary):
    '''
    Parameters
    ----------
    topics : list of list of str
        List of tokenized topics.
        Be careful to keep topics in the same order 
        as in your guidedlda model.

    bows : iterable of list of (int, number)
        Corpus in BoW format.

    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Gensim dictionary mapping of id word to create corpus.

    Returns
    -------
    coh_score : float
        average topics coherence of the whole model

    coh_topics : list 
        per-topic coherence scores (same order as topics)
    '''

    cm = CoherenceModel(
        topics=topics,
        corpus=bows,
        dictionary=dictionary,
        coherence='u_mass'
    )

    coh_score = cm.get_coherence()
    coh_topics = coherence_model.get_coherence_per_topic()

    return coh_score, coh_topics


def grid_search_lda_SED(texts, seed_topic_list,
                        n_topics_range, priors_range,
                        out_dir,
                        vectorizer_type='count',
                        iterations=2000,
                        verbose=True):
    '''
    Fit many topic models to pick the most tuned hyperparameters.
    Guidedlda version.

    Each fitted model is saved, filename being in the following format:
    {number of topics}T_{iteration rank}I_.{file extension}


    Parameters
    ----------
    texts : iterable
        already preprocessed text data you want to build seeds on.

    seed_topic_list : list of lists
        list of words, where in seed_topic_list[x][y] 
        x is a topic and y a word belonging in that topic.

    n_topics_range : iterable of int | int
        Number of topics to fit the model with.
        When fitting a single model, :int: is enough.
        Otherwise, input list of ints, a range, or other iterables.

    priors_range : tuple
        where 
        priors_range[0] is range for alpha 
        priors_range[1] is range for eta.

    out_dir : str
        path to a directory, where results will be saved (in a child directory).

    vectorizer_type : str, optional (default: 'count')
        Map documents using raw counts, or tfidf?
        Options = {"count", "tfidf"}

    iterations : int, optional (default: 2000)
        maximum number of iterations to fit a topic model with.

    verbose : bool, optional (default: True)
        print progress comments.


    Exports
    -------
    out_dir/report_lines/*
        pickled dict with model information
        (n topics, model coherence, per-topic coherence, hyperparameters)
        
    out_dir/models/*
        gensim objects, where the model is saved.
        
    out_dir/plots/*
        pyLDAvis visualizations of the model
    '''
    # INITIALIZATION
    # prepare foldrs
    make_folders(out_dir)

    # paths
    report_dir = os.path.join(out_dir, "report_lines", "")
    model_dir = os.path.join(out_dir, "models", "")
    plot_dir = os.path.join(out_dir, "plots", "")

    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(n_topics_range, int):
        n_topics_range = [n_topics_range]

    # PREPARE DATA
    # for guidedlda fiting
    X, seed_priors, vectorizer = init_guidedlda(
        texts=texts,
        seed_topic_list=seed_topic_list,
        vectorizer_type=vectorizer_type
    )

    # for coherence counting
    bows, dictionary = gensim_format(texts)

    # TRAIN MODELS
    for n_top in chain(n_topics_range):

        # iterate over priors
        # TODO
        for alpha, eta in priors_range:

            start_time = time() # track time
            i += 1 # track iterations

            # paths for saving
            filename = str(n_top) + "T_" + str(i) + "I_"
            report_path = os.path.join(report_folder + filename + '.pickle')
            model_path = os.path.join(model_folder + filename + '.model')
            pyldavis_path = os.path.join(plot_folder + filename + '_pyldavis.html')

            # train model
            model = guidedlda.GuidedLDA(
                n_topics=n_top,
                n_iter=iterations,
                alpha=alpha, eta=eta,
                random_state=7, refresh=10
            )

            # TODO: iterate seed_confidence?
            model.fit(X, seed_topics=seed_priors, seed_confidence=0.10)

            # track time usage
            training_time = time() - start_time
            if verbose:
                print('    Time: {}'.format(training_time))

            # save priors
            # TODO
            alpha = 'TODO'
            eta = 'TODO'

            # coherence
            coh_score, coh_topics = coherence_guidedlda(
                bows=bows, dictionary=dictionary
            )

            # save report
            report = (n_top, alpha, eta, training_time, coh_score, coh_topics)
            with open(report_path, 'w') as f:
                ndjson.dump(report, f)

            # save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # produce a visualization
            nice = pyLDAvis.sklearn.prepare(model, X, vectorizer)
            pyLDAvis.save_html(nice, pyldavis_path)

    return None
