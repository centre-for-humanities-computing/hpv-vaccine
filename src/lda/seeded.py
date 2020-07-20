'''
TODO
- everything
'''
import os
from itertools import chain
from time import time

from six.moves import cPickle as pickle

from sklearn.feature_extraction.text import CountVectorizer
import guidedlda
import pyLDAvis.sklearn

from src.utility.general import make_folders


def get_seeds(texts, seed_topic_list, vectorizer):
    '''
    Prepare data & seed priors for guidedlda.
    (Vectorize texts and extract hashed seeds)

    No more preprocessing is done here.


    Parameters
    ----------
    texts : iterable
        already preprocessed text data you want to build seeds on.

    seed_topic_list : list (of lists)
        list of words, where in seed_topic_list[x][y] 
        x is a topic and y a word belonging in that topic.

    vectorizer : str
        Map documents using raw counts, or tfidf?
        Options = {"count", "tfidf"}
    '''
    # texts are already preprocessed, so vectorizer gets this as tokenizer
    def do_nothing(doc):
        return doc

    # choose vectorizer
    if vectorizer is 'tfidf':
        # TODO: this is dummy code
        vectorizer = TfIdfVectorizer(
            analyzer='word',
            tokenizer=do_nothing,
            preprocessor=None,
            lowercase=False
        )

    elif vectorizer is 'count':
        vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=do_nothing,
            preprocessor=None,
            lowercase=False
        )

    else:
        raise ValueError("vectorizer: choose tfidf, or count")

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


def iterate_guidedlda(X, seed_priors, vect_obj,
                      n_topics_range, priors_range,
                      out_dir,
                      verbose=True):
    '''
    Fit many topic models to pick the most tuned hyperparameters.
    Guidedlda version.


    Parameters
    ----------
    X : ???
        vectorized data to fit the model with.

    seed_priors : ???
        array of hashed seeds to be feeded as priors for the model.

    vect_obj : ???
        vectorizer pipeline object
        Used for pyLDAvisualization.

    out_dir : str
        path to a directory, where results will be saved (in a child directory).


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

    # iterate over n topics
    report_list = []
    for n_top in chain(n_topics_range):

        # iterate over priors
        # TODO
        smallio = []
        for alpha, eta in chain(priors_range):

            start_time = time() # track time
            i += 1 # track iterations

            # paths for saving
            filename = str(n_top) + "T_" + str(i)
            report_path = os.path.join(report_folder + filename + '.pickle')
            model_path = os.path.join(model_folder + filename + '.model')
            pyldavis_path = os.path.join(plot_folder + filename + '_pyldavis.html')

            # train model
            model = guidedlda.GuidedLDA(
                n_topics=n_top,
                n_iter=2000,
                alpha=alpha, eta=eta,
                random_state=7, refresh=10
            )

            # TODO: iterate seed_confidence?
            model.fit(X, seed_topics=seed_priors, seed_confidence=0.10)

            # track time usage
            training_time = time() - start_time
            if verbose:
                print('    Time: {}'.format(training_time))

            # TODO: coherence
            # coh_score : float : whole modle score
            # coh_topics : list : per-topic coherence

    #         if verbose:
    #             print('    Coherence: {}'.format(coh_score.round(2)))

            # save report
            # TODO: fuck pickle, let's ndjson
            report = (n_top, alpha, eta, training_time, coh_score, coh_topics)
            report_list.append(report)
            with open(report_path, 'wb') as f:
                pickle.dump(report, f)

            # save model
            # TODO: sklearn
            model.save(model_path)

            # produce a visualization
            nice = pyLDAvis.sklearn.prepare(model, X, vect_obj)
            pyLDAvis.save_html(nice, pyldavis_path)

    return None


def grid_search_lda_SED(texts, seed_topic_list, vectorizer):
    
    X, seed_priors, vect_obj = get_seeds(texts, seed_topic_list, vectorizer)
    
    iterate_guidedlda(X, seed_priors, vect_obj,
                      n_topics_range, priors_range
                      report_folder, model_folder, plot_folder,
                      verbose=True)
    return None