'''Novelty, Transience & Resonance of your document-topic matrix of choice.

TODO
----
- calculate_ntr: IndexError
- kz: seems like it's only working 50%
'''
import os
from itertools import chain

import ndjson
import pandas as pd

from gensim.models import LdaModel
from gensim.corpora import Dictionary

from src.topicevolution.infodynamics import InfoDynamics
from src.topicevolution.entropies import jsd


def summarzie_doc_top():
    '''
    Summarizes probability distributions from a document-topic matrix
    into a distribution, corresponding to chunks of certain temporal length.

    E.g. get a probability distribution of 10 minutes of facebook posts.

    NOTE: maybe not even needed in HPV, but deffinitely yes in HOPE!

    Parameters
    ----------
    ???
    '''
    return None


def kz(df, window, iterations):
    """KZ filter implementation
    
    pd.rolling_mean() which the OG code relied on is deprecated.
    Rolling.mean() probably doen't do the same thing with the iterations.
    
    Parameters
    ----------
    df : pd.DataFrame | pd.Series

    window : int 
        filter window m in the units of the data (m = 2q+1)

    iterations : int
        the number of times the moving average is evaluated
    
    Source: https://stackoverflow.com/questions/32788526/python-scipy-kolmogorov-zurbenko-filter
    """
    z = df.copy()
    for i in range(iterations):
        z = df.rolling(window=window, min_periods=1, center=True).mean()
    return z


def calculate_ntr(doc_top_prob, ID, window, out_dir=None):
    '''Based on document-topic matrix, calcualte Novelty, Transience & Resonance.
    
    Parameters
    ----------
    doc_top_prob : list/array (of lists)
        document-topic-matrix of your topic model

    ID : list/array
        identifier, or coordinate for each document (identical order as data)
        Required by InfoDynamics

    window : int or range
        Number of documents to calculate Nov, Tra, Res against.
        See Barron, Huang, Spang & DeDeo (2018) for details.

    out_dir : str (optional)
        Directory where to save the results.
    '''
    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(window, int):
        window = [window]

    # make sure there is a folder to save it
    if out_dir:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    for w in chain(window):
        # signal calculation
        idmdl = InfoDynamics(data=doc_top_prob, time=ID,
                             window=w, weight=0, sort=False)
        idmdl.novelty(meas = jsd)
        idmdl.transience(meas = jsd)
        idmdl.resonance(meas = jsd)

        lignes = list()
        for i, time in enumerate(ID):
            d = dict()
            d["date"] = time
            # HACK because of IndexError
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

        if out_dir:
            # make a filename
            filename = str(w) + 'W' + '.csv'
            outpath = out_dir + filename

            # export
            with open(outpath, "w") as f:
                ndjson.dump(lignes, f)

    return None
