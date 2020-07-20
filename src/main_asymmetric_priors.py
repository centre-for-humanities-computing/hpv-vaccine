"""
Fit a non-seeded topic model & extract topic evolution.

TODO:
in _calcualte_ntr():
    save document-topic matrix as csv
    make it possile to run multiple models at once
    
in main():
    makefile style checking for what has been run already.
"""
import argparse
import os

import pandas as pd
import ndjson

import text_to_x as ttx # TODO: work around
from gensim import models, corpora

from gridsearch.lda_gensim import lda_grid_search_ASM
from topicevolution.infodynamics import InfoDynamics
from topicevolution.entropies import jsd


def _preprocess2(input_path, out_dir=None):
    '''Run gensim phrase detection, remove empty, keep dates.
    Returns lists of tokens & dates
    
    Specific method for Covid-19 Twitter megafiles.
    
    Parameters
    ----------
    input_path : str
        Assuming a single, large ndjson file that has _already_ been preprocessed.
        (i.e. tokenization, lemmatization & feature selection)
        
    out_dir : str (optional)
        path to a directory, where results will be saved (in a child directory).
    '''
    # import
    with open(input_path) as f:
        tweets_raw = ndjson.load(f)

    # extract data of interest
    mega_dfs = [pd.DataFrame(dat['tokens']) for dat in tweets_raw]
    lang = [dat['lang'] for dat in tweets_raw]
    dates = [dat['created_at'] for dat in tweets_raw]

    # convert to a nice format
    # keep only "meaningful" POS
    # (i.e. noun, propnoun, adj, verb, adverb)
    ttt_wrap = ttx.TextToTokensWrapper(mega_dfs, lang)
    mega_ttx = ttx.TextToTopic(ttt_wrap, tokentype='lemma')

    # initialize phrase detection
    phrases = models.Phrases(mega_ttx.tokenlists, delimiter=b" ")
    # find phrases
    phraser = models.phrases.Phraser(phrases)
    # extract texts with phrases detected
    phrase_list = [phraser[tl] for tl in mega_ttx.tokenlists]

    # missing any data?
    assert len(dates) == len(mega_ttx.tokenlists) == len(phrase_list)

    # put together dates and tweets
    phrase_doc = []
    for (i, tweet) in enumerate(phrase_list):
        d = dict()
        d['date'] = dates[i]
        d['text'] = tweet
        phrase_doc.append(d)

    # remove empty tweets
    phrase_doc = [doc for doc in phrase_doc if doc['text']]
    texts = [doc['text'] for doc in phrase_doc]

    # if saving enabled
    if out_dir:
        data_dir = os.path.join(out_dir, "data", "")
        # create a child directory, if it doesn't exist yet
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)   
        # save it
        bows_path = os.path.join(data_dir, 'phrased_bows.ndjson')
        with open(bows_path,'w') as f:
            ndjson.dump(phrase_doc, f)

    return texts, dates


def _fit_lda(out_dir, texts, n_topics, return_corpus=False):
    '''Wrapper of gridsearch.lda_gensim.lda_grid_search_ASM

    Fit topic models and search for optimal hyperparameters.
    Asymmetrical priors alpha & eta are guessed automatically.


    Parameters
    ----------
    out_dir : str
        path to a directory, where results will be saved (in a child directory).

    texts : e.g. list
        tokenized documents in texts[i][j] format,
        where i leads to documents and j to tokens.
        Probably works with some other formats as well.

    n_topics : int
        number of LDA topics.
        If you are not calculating NTR afterwards, 
        can also be multiple numbers, i.e. list of ints, or a range.


    Exports
    -------
    report_lines/*
        pickled dict with model information
        (n topics, coherence, hyperparameters)
        
    models/*
        gensim objects, where the model is saved.
        
    plots/*
        pyLDAvis visualizations of the model
    '''
    # input texts to gensim format
    dictionary = corpora.Dictionary(texts)
    bows = [dictionary.doc2bow(tl) for tl in texts]

    # get output paths
    report_dir = os.path.join(out_dir, "report_lines", "")
    model_dir = os.path.join(out_dir, "models", "")
    plot_dir = os.path.join(out_dir, "plots", "")

    # create folders
    for outdir in [report_dir, model_dir, plot_dir]:
        # check if dir already exists
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    # fit lda
    # prints comments
    lda_grid_search_ASM(texts=texts,
                        dictionary=dictionary,
                        bows=bows,
                        n_topics_range=n_topics,
                        report_folder=report_dir,
                        model_folder=model_dir,
                        plot_folder=plot_dir,
                        verbose=True)

    if return_corpus:
        # TODO: possible to run multiple models at once, when runing _calculate_ntr()
        if isinstance(n_topics, int):
            lda_model_path = os.path.join(model_dir, str(n_topics) + 'T.model')
            return lda_model_path, bows
        else:
            ValueError('If return_corpus=True, n_topics must be a single int')


def _calculate_ntr(lda_model_path, bows, dates, window, out_dir):
    '''Based on document-topic matrix, calcualte Novelty, Transience & Resonance.
    
    Parameters
    ----------
    lda_model_path : str
        path to gensim *.model file containing a trained LDA model.

    bows : dd
        dd

    date : dd
        dd

    window : int
        number of documents to calculate Nov, Tra, Res against.
        See Barron, Huang, Spang & DeDeo (2018) for details.

    out_dir : str
        path to a directory, where results will be saved (in a child directory).
    '''
    # load model
    model = models.LdaModel.load(lda_model_path)
    
    # GENSIM doc-top matrix
    # min topic prob = 0 for uniform array shape
    doc_top = [model.get_document_topics(doc, minimum_probability=0)
               for doc in model[bows]]

    # unnest (n topic, prob) tuples
    doc_top_prob = [
        [prob for i, prob in doc]
        for doc in doc_top
    ]

    # TODO: export document-topic matrix
#     doctopics_df = pd.DataFrame([])
#     for doc in doc_top_prob:
#         doc = pd.DataFrame(doc).transpose() 
#         doctopics_df = doctopics_df.append(doc.iloc[1,:],
#                                            ignore_index=True)
#     doc_top_matrix_path = os.path.join(
#         out_dir, 'data',
#         lda_model_path.replace('.model',''), "T_doctopmat.csv"
#     )
#     doctopics_df.to_csv('topic_models/batch_200621/topic_evolution/22T_doctop.csv')

    # signal calculation
    idmdl = InfoDynamics(data=doc_top_prob, time=dates,
                         window=window, weight=0, sort=False)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    lignes = list()
    for i, time in enumerate(dates):
        d = dict()
        d["date"] = time
        # HACK
        try:
            d["novelty"] = idmdl.nsignal[i] 
            d["transience"] = idmdl.tsignal[i]
            d["resonance"] = idmdl.rsignal[i]
            d["nsigma"] = idmdl.nsigma[i]
            d["tsigma"] = idmdl.tsigma[i]
            d["rsigma"] = idmdl.rsigma[i]
        except IndexError:
            print("[info] there was an Index Error, but we're taking care of the situation")
            pass
        lignes.append(d)

    # save
    n_topics = (lda_model_path
                # e.g. */models/10T.model -> 10T.model
                .replace(out_dir+'models/', '')
                # e.g. 10T.model -> 10T
                .replace('.model','')
               )
    filename = str(n_topics) + '_' + str(window) + 'W_NTR.ndjson' # so no T, but yes W
    with open(os.path.join(out_dir, 'data', filename), "w") as f:
        ndjson.dump(lignes, f)

    return None


def main():
    '''Ruin everything.

    TODO: makefile style checking for what has been run already.
    '''
    # initialize argparser with a desc
    ap = argparse.ArgumentParser(
        description="Fit non-seeded topic models & extract topic evolution"
    )
    ap.add_argument("-p", "--inpath",
                   required=True,
                   help="path to input data")
    # output dir
    ap.add_argument("-o", "--outpath",
                    required=True,
                    help="path to a directory, where results will be saved (in a child directory)")
    # number of LDA topics
    ap.add_argument("-n", "--ntopics",
                    required=False,
                    type=int, default=10,
                    help="number of LDA topics for the model")
    # window
    ap.add_argument("-w", "--window",
                    required=False,
                    type=int, default=3,
                    help="window to compute novelty and resonance over")
    # parse that
    args = vars(ap.parse_args())

    # run functions down the line
    print('[info] Preprocessing {}'.format(args['inpath']))
    texts, dates = _preprocess2(
        input_path=args['inpath'],
        out_dir=args['outpath']
    )

    print('[info] Training LDA on {} topics'.format(args['ntopics']))
    lda_model_path, bows = _fit_lda(
        out_dir=args['outpath'],
        texts=texts,
        n_topics=args['ntopics'],
        return_corpus=True
    )

    print('[info] Calculating Nov, Tra & Res in a window of {} documents'.format(args['window']))
    _calculate_ntr(
        lda_model_path=lda_model_path,
        bows=bows,
        dates=dates,
        window=args['window'],
        out_dir=args['outpath']
    )


if __name__=="__main__":
    main()
