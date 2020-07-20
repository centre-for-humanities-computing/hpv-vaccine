"""Iterate Gensim's LDA in search of good hyperparameters.
"""

import os
from itertools import chain
from time import time

from six.moves import cPickle as pickle

from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaModel

import pyLDAvis.gensim


def lda_grid_search_ASM(texts, dictionary, bows, n_topics_range,
                        report_folder, model_folder, plot_folder,
                        verbose=True):
    '''Fit topic models and search for optimal hyperparameters.

    LDA will be fitted for each number of topics,
    returned will be the model, it's coherence score and
    corresponding _asymmetrical_ priors the model learned (alpha and eta)


    Parameters
    ----------
    texts : list
        preprocessed corpus, where texts[0] is a document
        and texts[0][0] is a token.

    dictionary : gensim.corpora.Dictionary
        gensim dictionary of texts

    bows : gensim.doc2bow
        bag of words representation of the corpus (texts)

    n_topics_range : range of int
        range of integers to use as the number of topics
        in interations of the topic model.

    report_folder : str
        path to folder, where report objects going to be dumped

    model_folder : str
        path to folder to save models

    plot_folder : str
        path to folder to save plots


    Exports
    -------
    report_lines/*
        pickled dict with model information
        (n topics, model coherence, per-topic coherence, hyperparameters)
        
    models/*
        gensim objects, where the model is saved.
        
    plots/*
        pyLDAvis visualizations of the model
    '''
    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(n_topics_range, int):
        n_topics_range = [n_topics_range]

    # iterate
    report_list = []
    for n_top in chain(n_topics_range):

#         if verbose:
#             print("{} topics".format(n_top))

        start_time = time()

        # paths for saving
        report_path = os.path.join(report_folder + str(n_top) + 'T.pickle')
        model_path = os.path.join(model_folder + str(n_top) + 'T.model')
        pyldavis_path = os.path.join(plot_folder + str(n_top) + 'T_pyldavis.html')

        # train model
        model = LdaModel(
            corpus=bows,
            iterations=50,
            ## optimizing hyperparameters
            num_topics=n_top,
            alpha='auto',
            eta='auto',
            ## bizzare hyperparameters
            decay=0.5,
            offset=1.0,
            eval_every=10,
            gamma_threshold=0.001,
            minimum_probability=0.01,
            minimum_phi_value=0.01,
            ## utility
            random_state=None,
            per_word_topics=False,
            id2word=dictionary,
            passes=1)

        # track time usage
        training_time = time() - start_time
        if verbose:
            print('    Time: {}'.format(training_time))

        # coherence
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            corpus=bows
        )

        coh_score = coherence_model.get_coherence()

        if verbose:
            print('    Coherence: {}'.format(coh_score.round(2)))

        coh_topics = coherence_model.get_coherence_per_topic()

        # save priors
        alpha = model.alpha[0]
        eta = model.eta[0]

        # save report
        report = (n_top, alpha, eta, training_time, coh_score, coh_topics)
        report_list.append(report)
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)

        # save model
        model.save(model_path)

        # produce a visualization
        vis = pyLDAvis.gensim.prepare(model, bows, dictionary,
                                      sort_topics=False)
        pyLDAvis.save_html(vis, pyldavis_path)

        return None
